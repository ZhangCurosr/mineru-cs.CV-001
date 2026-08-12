# A second-order theory of texture for depth from focus

Sreekar Sai Ranganathan and Ioannis Gkioulekas

Carnegie Mellon University, Pittsburgh PA, USA {ssrangan,igkioule}@andrew.cmu.edu

Abstract. We present a theory of textured appearance of optically rough surfaces based on wave optics, emphasizing the role of texture for passive depth from focus. Our theory shows that even surfaces that traditional computer vision would consider textureless can produce textured appearance, due to subjective speckle from surface microgeometry. We analyze the properties of this second-order texture, and show that we can enhance its contrast under natural ambient lighting by simply using a narrowband spectral filter. Doing so results in dramatic improvements in passive depth reconstruction of seemingly textureless scenes, as we demonstrate through extensive theory, simulations, and real-world experiments.

Keywords: passive depth · texture · wave optics · depth from focus

## 1 Introduction

Are textureless scenes recoverable? Sundaram and Nayar [1] posed this fundamental question to the computer vision community nearly three decades ago. Their analysis suggested that little geometrical information exists in images of a textureless scene taken by a standard camera, thus greatly hindering passive depth sensing of such scenes. In the succeeding years, computer vision has made great strides towards developing methods for passive depth sensing, leveraging focus or parallax cues that can be inferred from texture even under uncontrolled ambient illumination. Despite these advances, the assumption has remained that passive depth sensing of textureless scenes is dificult, if not impossible. In this paper, we challenge this assumption: we show that merely adding a narrowband spectral filter in front of an otherwise standard lens-based camera allows using depth from focus (DFF) to recover textureless objects (Figure 1).

Understanding this surprising finding requires examining another fundamental question in computer vision: What is texture? The conventional definition requires spatial variations of surface reflectance or normal at a scale resolvable by the camera at the working imaging magnification, manifesting as spatially varying intensities in focused images of the surface. Thus, a surface with spatially constant bidirectional reflectance distribution function (BRDF)—the macroscopic description of reflectance—would be considered “textureless”.

We challenge this definition by revisiting classical surface appearance models based on wave optics. Our analysis predicts that even such “textureless” surfaces will produce non-constant in-focus images showing subjective speckle—high frequency intensity variations due to difraction at surface microgeometry that the camera cannot resolve. Understanding this efect requires analysis of secondorder intensity statistics, instead of just the mean (first-order moment) intensity predicted by classical appearance models. We term this efect second-order texture, to distinguish it from the classical first-order texture.

![](images/13d3f1466df920dc9e8f9c0eafa4a4a604df9aa95b5dbcbf002b1e0cbf29123b.jpg)  
Figure 1: Computer vision traditionally considers depth recovery of scenes with sparse texture features impossible with passive methods such as depth from focus (top). We show that simply adding a narrowband spectral filter to the camera makes passive depth recovery of such scenes possible (bottom). We explain this surprising finding by developing a theory of second-order texture—a form of subjective speckle whose in-focus contrast is enhanced with decreasing spectral bandwidth (crops to the left).

Second-order texture will be useful for DFF only if it has high-enough contrast to overcome sensor noise. Prior work assumes that significant subjective speckle is possible only under coherent illumination or at microscopic magnifications. We show that using a narrowband spectral filter (10–100 nm) results in high-contrast second-order texture under typical passive computer vision conditions—outdoor and indoor ambient lighting, macroscopic magnifications. Using such a filter also reduces incident flux, necessitating increased exposure time to maintain signal-to-noise ratio (SNR). We show that, even under severe underexposure (3–4 stops), second-order texture is discernible from noise. On the whole, using a narrowband spectral filter can dramatically improve DFF performance (Figure 2).

To present our findings, first, we provide background on DFF, and characterize depth recoverability conditions as a function of texture contrast and SNR (§ 3). Next, we develop our second-order theory of texture, and analyze the factors impacting the contrast of second-order texture (§ 4). Then, we use this analysis to assess, through theory and simulations, how DFF performance varies with filter spectral bandwidth and exposure time (§ 5). Last, we perform DFF experiments under ambient indoor and outdoor illumination that validate our findings (§ 6). The supplemental PDF includes all proofs and additional experimental results. The project website<sup>1</sup> includes interactive visualizations, code, and datasets.

## 2 Related work

Passive depth sensing in computer vision. Passive methods use parallax or focus to infer depth under ambient illumination. Parallax methods such as stereo [2–5] infer depth by triangulating correspondences across multi-view images. Among focus methods, depth from focus [6–11] infers depth by assessing per-pixel focus in a dense focal stack. Depth from defocus [12–16] uses two images at diferent focus settings to estimate defocus blur and thus depth. Other methods change both focus and aperture (confocal stereo [17]), vary focus diferentially (focal flow [18, 19]), or engineer high-frequency defocus blur (coded aperture [20–24]). Focus methods can be analyzed as stereo with a baseline equal to the lens aperture [25], a relationship used in lightfield [26, 27] and dual-pixel [28–31] methods. Unfortunately, these methods fundamentally require texture, to either establish correspondence or assess focus, and are unreliable in textureless scenes.

![](images/36fa0fbc8d4304b45f92ecdce99e3f48f34f397980a0650dab2df7f3fb76db6d.jpg)  
Figure 2: We demonstrate fine-scale depth recovery of objects with sparse texture features under ambient lighting indoors (ceiling lights, left) and outdoors (sunlight, right). In both cases, adding a spectral filter dramatically improves DFF performance.

Passive interferometric imaging. Several methods leverage the weak coherence of ambient illumination through interferometry—correlation measurements of superimposed light waves—for passive 3D imaging tasks. Incoherent digital holography methods take interferometric measurements to create holographic scene representations [32–37]. Other such methods can perform 3D localization for occluded imaging [38–42]. Closer to our work, interferometry has been used for passive depth sensing—through focus [43], parallax [44], or time-of-flight [45]—of even textureless scenes. Unfortunately, these methods require complex and sensitive optical systems, limiting their practical utility. Like these methods, ours achieves textureless passive depth sensing by leveraging weak coherence efects—subjective speckle from ambient illumination—but it does so through just a very simple modification to a standard depth-from-focus system—mounting a narrowband spectral filter in front of the lens.

Speckle imaging. Speckle has been extensively studied in optics since the invention of the laser. We refer to dedicated textbooks [46–48], and limit our discussion to literature in vision and graphics. Computer vision methods have used speckle for tamper detection [49], motion tracking [50–52], vibrometry [53, 54], imaging around corners [55], and through occluders [56–58]. All these methods are active, introducing coherent illumination into the scene to induce speckle. These applications have also motivated work on speckle rendering [59–63]. Lastly, speckle modeling is at the foundation of classical appearance models for rough surfaces [64–67], a foundation we revisit to develop our theory.

## 3 Depth from focus & recoverability conditions

We begin with background on depth from focus (DFF) and an analysis of recoverability conditions, focusing on the role of texture and sensor noise. DFF uses a focal stack of images, captured by stepping the focusing depth of the camera lens. We denote this stack as $\{ \widetilde { \mathrm { I } } _ { j } ( \overrightarrow { p } ) \} _ { j = 1 } ^ { J }$ , where $\widetilde { \mathrm { I } } _ { j } ( \overrightarrow { p } )$ is the sensor’s intensity measurement at the pixel $\overrightarrow { p }$ e eand focus index j—we use 2D vectors $\overrightarrow { p }$ to indicate pixel center locations, and tildes to highlight quantities impacted by sensor noise, such as the noisy intensity I and its functionals. Depth recovery requires determining for each pixel $\overrightarrow { p }$ ethe (most) in-focus index $j _ { \mathrm { i f } } ( \overrightarrow { p } )$

DFF estimates $j _ { \mathrm { i f } } ( \vec { p } )$ by finding the image that maximizes a measure of spatial intensity variation (or “sharpness”) for a patch $\mathrm { P } _ { T } ( \overrightarrow { p } )$ of $T$ pixels $( e . g .$ ${ \sqrt { T } } \times { \sqrt { T } } { \mathrm { - s q u a r e } } )$ around $\overrightarrow { p } .$ . Several such focus measures are available [68]; we consider the squared sample coeficient of variation $\widetilde { \mathrm { c } } _ { j } ^ { 2 } ( \overrightarrow { p } ) : = \widetilde { \mathrm { s } } _ { j } ^ { 2 } ( \overrightarrow { p } ) \big / \widetilde { \mathrm { m } } _ { j } ^ { 2 } ( \overrightarrow { p } )$ ecomputed from the sample mean and sample variance of gray patch intensities,

$$
\widetilde { \mathfrak { m } } _ { j } ( \vec { p } ) : = \frac { 1 } { T } \sum _ { \vec { p } ^ { \prime } \in \operatorname { P } _ { T } ( \vec { p } ) } \widetilde { \mathrm { I } } _ { j } ( \vec { p } ^ { \prime } ) , \quad \widetilde { \mathfrak { s } } _ { j } ^ { 2 } ( \vec { p } ) : = \frac { 1 } { T - 1 } \sum _ { \vec { p } ^ { \prime } \in \operatorname { P } _ { T } ( \vec { p } ) } \Bigl ( \widetilde { \mathrm { I } } _ { j } ( \vec { p } ^ { \prime } ) - \widetilde { \mathrm { I } } _ { j } ( \vec { p } ) \Bigr ) ^ { 2 } .\tag{1}
$$

We use $\widetilde { \mathrm { c } } _ { j } ^ { 2 } ( \overrightarrow { p } )$ because it simplifies analysis, but our findings extend to other ecommon focus measures $( \ S \ G )$ . DFF estimates $j _ { \mathrm { i f } } ( \overrightarrow { p } )$ as $j ^ { \star } ( \vec { p } ) : = \mathrm { a r g m a x } _ { j } \widetilde \mathrm { c } _ { j } ^ { 2 } ( \vec { p } )$ We refer to $\widetilde { \mathrm { c } } _ { j } ^ { 2 } ( \overrightarrow { p } )$ as simply sample contrast, and omit $\overrightarrow { p }$ ewhere convenient.

eThe justification for estimating $j _ { \mathrm { i f } } \ \mathrm { a s } \ j ^ { \star }$ is that the contrast $\widetilde { \mathrm { c } } _ { j } ^ { 2 }$ will be maximal eat the in-focus index because defocus acts as a low-pass filter. This justification is valid under two conditions [9, 68]:

1. Texture condition: The in-focus patch $\mathrm { P } _ { T } ( \overrightarrow { p } )$ has appreciable texture, $i . e . ,$ spatial intensity variations, in the absence of noise.

2. Signal-to-noise ratio condition: The sensor noise is suficiently low so that intensity variations due to texture are discernible from those due to noise.

These conditions are afected by scene appearance (surface reflectance, lighting), imaging conditions (lens aperture, exposure, sensor noise), and algorithmic choices (patch size $T )$ . In the rest of this section, we characterize the impact and interplay of these factors. We first distinguish between the measured noisy intensities $\widetilde { \mathrm { I } } _ { j }$ and the corresponding noise-free intensities $\mathrm { I } _ { j } .$ e, which we define formally below. We also define the noise-free contrast, sample variance, and mean $\mathrm { c } _ { j } ^ { 2 } , \mathrm { s } _ { j } ^ { 2 }$ , and m<sub>j</sub> (respectively) analogously, and use them to define a notion of recoverability.

## Definition 1: α-recoverability

For any $\alpha \in [ 0 , 1 ]$ , we say that the in-focus index $j _ { \mathrm { i f } }$ is α-recoverable when $\mathrm { P r } ( \widetilde { \mathrm { c } } _ { \mathrm { i f } } ^ { 2 } < \widetilde { \mathrm { c } } _ { \mathrm { o o f } } ^ { 2 } ) \stackrel { \cdot } { \le } \dot { \alpha }$ , where: $1 . \ \widetilde { \mathrm { c } } _ { \mathrm { i f } } ^ { 2 }$ is the focus measure at $j _ { \mathrm { i f } } .$ , and $2 . { \ \widetilde { \mathrm { c } } } _ { \mathrm { o o f } } ^ { 2 }$ is the e e efocus measure at an out-of-focus index $j _ { \mathrm { o o f } } \neq j _ { \mathrm { i f } }$ ewith suficient defocus such that noise-free intensities are nearly identical, thus $\mathrm { c } _ { \mathrm { o o f } } ^ { 2 } \approx 0$

We use α-recoverability as a weaker, but analytically tractable, proxy of true recoverability to formalize the interplay of texture and noise (Proposition 2).

We next characterize the statistics of noise-free and noisy intensities at $j _ { \mathrm { i f } }$ and $j _ { \mathrm { o o f } }$ denoted as $\operatorname { I } _ { \mathrm { i f } }$ and $\operatorname { I } _ { \mathrm { o o f } }$ for the noise-free, and analogously for the noisy case. Texture model. The in-focus noise-free intensities $\operatorname { I } _ { \mathrm { i f } } ( { \overrightarrow { p } } )$ are themselves random variables described by a statistical texture model. For weakly textured surfaces, as we define them in $\ S \ 4 . 2$ , it sufices to consider the mean $\mu _ { \mathrm { I } }$ and variance $\sigma _ { \mathrm { { I } } } ^ { 2 }$ of their stationary pointwise distribution. From the definition of $j _ { \mathrm { o o f } }$ in Definition 1, the noise-free intensities $\mathrm { I _ { o o f } }$ will all equal $\mu _ { \mathrm { I } }$ and have zero variance.

Noise model. We can relate noisy to noise-free intensities at both $j \in \{ j _ { \mathrm { i f } } , j _ { \mathrm { o o f } } \}$ using the standard Poisson–Gaussian model $[ 6 9 ] , \widetilde { \mathrm { I } } _ { j } = \operatorname* { m i n } ( \mathrm { I } _ { j } + n _ { j } , \mathrm { I } _ { \operatorname* { m a x } } )$ , where: $1 . \ \mathrm { I } _ { j }$ is proportional to the incident flux $\Phi _ { j }$ eand exposure time $t , \mathrm { I } _ { j } : = \Phi _ { j } t \zeta / g ,$ implying that $\mu _ { \Phi } : = \mathbb { E } [ \Phi ] = \mu _ { \mathrm { I } } g / \varsigma . 2$ . The noise term $n _ { j }$ is a random variable combining photon shot (Poisson) noise and read (Gaussian) noise. It has mean $\mathbb { E } [ n _ { j } \mid \mathrm { I } _ { j } ] = 0$ and signal-dependent variance $\mathbb { V } [ n _ { j } \mid \mathrm { I } _ { j } ] = \mathrm { I } _ { j } { 1 / { g } } + \sigma _ { \mathrm { r e a d } \ne j } ^ { 2 }$ , where $\sigma _ { \mathrm { r e a d } } ^ { \bar { 2 } } : = \sigma _ { \mathrm { p r e } } ^ { 2 } + g ^ { 2 } \sigma _ { \mathrm { p o s t } } ^ { 2 } . 3$ . The constant ζ depends on quantum eficiency, g on ISO, $\sigma _ { \mathrm { p r e } }$ and $\sigma _ { \mathrm { p o s t } }$ on readout circuitry, and $\mathrm { I } _ { \mathrm { m a x } }$ on the saturation limit. This model assumes for simplicity no dark current or discretization, assumptions we revisit in $\ S 5 .$ . The marginal variance of noise $n _ { j }$ thus equals $\sigma _ { n } ^ { 2 } : = \mathbb { V } [ n _ { j } ] = \mu _ { \mathrm { I } } / _ { g } + \sigma _ { \mathrm { r e a d } } ^ { 2 } / g ^ { 2 }$ using the fact that the noise-free mean intensity equals $\mu _ { \mathrm { I } }$ at j<sub>if</sub> and $j _ { \mathrm { o o f } }$

Recoverability condition. Using these texture and noise models, we can define:

$$
\kappa _ { \mathrm { I } } ^ { 2 } : = \frac { \sigma _ { \mathrm { I } } ^ { 2 } } { \mu _ { \mathrm { I } } ^ { 2 } } , \quad \kappa _ { n } ^ { 2 } : = \frac { \sigma _ { n } ^ { 2 } } { \mu _ { \mathrm { I } } ^ { 2 } } = \frac { 1 } { t \zeta \mu _ { \Phi } } \bigg ( 1 + \frac { \sigma _ { \mathrm { r e a d } } ^ { 2 } } { t \zeta \mu _ { \Phi } } \bigg ) .\tag{2}
$$

The squared texture contrast $\kappa _ { \mathrm { I } } ^ { 2 }$ and reciprocal squared SNR $\kappa _ { n } ^ { 2 }$ measure texture and noise strength. We can relate them to recoverability as follows.

Proposition 2: Necessary and suficient condition for α-recoverability For any $\alpha \in [ 0 , 1 ]$ , the in-focus index $j _ { \mathrm { i f } }$ is α-recoverable if and only if:

$$
\begin{array} { r l r } { \mathrm { t e x t u r e } } & { \mathrm { S N R } } & { \mathrm { p a t c h ~ s i z e } } \\ { \mathcal { Z } \left( \begin{array} { l l } { \kappa _ { \mathrm { I } } ^ { 2 } } & { . } & { \frac { 1 } { \kappa _ { n } ^ { 2 } } } \end{array} \right) \cdot } & { \sqrt { \frac { T - 1 } { 2 } } } & { > \phi ^ { - 1 } ( 1 - \alpha ) , } \end{array}\tag{3}
$$

where: 1. $\mathcal { Z } ( x ) : = \ d x / \sqrt { 1 + ( x + 1 ) ^ { 2 } }$ is a monotonically increasing function; and $2 . \phi ^ { - 1 } ( \cdot )$ is the inverse CDF of the standard normal distribution.

Proposition 2 (proved in supplement) provides an interpretable view of the three key factors impacting recoverability: texture contrast, SNR, and patch size.

The green highlighted factor in Equation (3) suggests improving recoverability by increasing SNR. Doing so is possible $\mathrm { b y }$ increasing exposure time t (Equation (2)), but only up to the saturation limit $\mathrm { I } _ { \mathrm { m a x } }$ . Further increasing SNR is possible by averaging captures from multiple sensor readouts. However, doing so increases acquisition time by more than the increase in total exposure time.

The brown highlighted factor in Equation (3) suggests improving recoverability by increasing the patch size T—representing the decreasing impact of noise in the sample contrast $\mathrm { \tilde { c _ { i f } ^ { 2 } } }$ as samples increase. However, increasing $T$ comes at the cost of decreased lateral resolution and increased errors close to depth discontinuities (which are not accounted for in standard DFF).

The classical understanding of texture in computer vision suggests that we cannot improve the blue highlighted factor in Equation (3) while maintaining passive operation— $\cdot i . e .$ , without active illumination to artificially texture surfaces. In what follows, we show this understanding to be incorrect: We present a secondorder theory of texture $( \ S \ 4 )$ based on wave-optical scattering of light, which predicts that it is indeed possible to amplify texture by simply adding a narrowband spectral filter in front of the camera. We provide extensive experimental verification $( \ S \ G )$ of this prediction, showing that the fundamental requirement for texture in passive depth recovery is not as severe as previously thought. Of course, adding a spectral filter also worsens the SNR factor in Equation (3), assuming fixed exposure time. We show (§ 5) that the improvement in texture contrast outweighs the worsening in SNR after only a sublinear increase in exposure time.

## 4 A second-order theory of texture

Further analysis of $\kappa _ { \mathrm { I } } ^ { 2 }$ in Equation (3) requires characterizing the pointwise distribution of pixel intensities $\operatorname { I } ( { \overrightarrow { p } } )$ . We assume for simplicity that the incident flux $\Phi _ { j } ( \vec { p } )$ at a pixel is proportional to the irradiance $\operatorname { E } ( { \overrightarrow { p } } )$ at its center, $\Phi _ { j } ( \vec { p } ) =$ $\mathrm { E } _ { j } ( \vec { p } ) \varDelta _ { \mathrm { p } } ^ { 2 }$ . It follows that $\bar { \kappa _ { \mathrm { I } } ^ { 2 } } = \kappa _ { \mathrm { E } } ^ { 2 } : = \sigma _ { E } ^ { 2 } / \mu _ { E } ^ { 2 }$ , thus it sufices to analyze $\kappa _ { \mathrm { { F } } } ^ { 2 }$

Our goal in this section is thus to characterize the irradiance $\operatorname { E } ( { \overrightarrow { p } } )$ . In $\ S \ 4 . 1$ we use wave-optical principles of light scattering to derive an expression of $\operatorname { E } ( { \overrightarrow { p } } )$ for a camera imaging optically rough surfaces under ambient illumination. We then explain how this expression relates to appearance models in computer vision [64–66, 70], and why the appearance of real-world rough surfaces is always textured due to subjective speckle, even for surfaces that classical computer vision would consider textureless. In $\ S \ 4 . 2$ we quantify the strength of this textured appearance as a function of imaging and illumination conditions.

## 4.1 First and second-order texture

To simplify exposition, and without loss of generality, we assume a thin-lens camera that images a planar, frontoparallel, and optically rough surface under far-field illumination (Figure 3). Our problem setting and derivation closely adapt Goodman [46, §§5.8 & 6.3]. We use a 3D coordinate system where the z axis is parallel to the optical axis and the macroscopic surface normal nˆ. We use 3D unit-norm vectors $\hat { \omega }$ for directions, and $\hat { \omega } _ { x y }$ for their projection on the x–y plane. We use 2D vectors $\overrightarrow { p }$ on the x–y plane interchangeably for both sensor and surface points, and identify each sensor point with the surface point it maps to through its central ray $\hat { v } \ \left( i . e . \right.$ , p

![](images/c72e728ed88e1e45d496dcf3445ef16fff6e8eda3177dc71d83430b78a57ce84.jpg)  
Figure 3: Problem setup and notation.

We model the camera lens using its focus-dependent point spread function $( \mathrm { P S F } ) \operatorname* { P } _ { \mathrm { c } } ( \vec { p } - \vec { q } )$ , and use $\mathrm { w } _ { \mathrm { c } } ( { k } )$ for the spectral sensitivity function (SSF) of the sensor. We model the far-field illumination as a spectral radiance environment map that we assume separable in direction and wavenumber, $\mathrm { L } _ { \mathrm { i } } ( \hat { \omega } , \boldsymbol { k } ) : = \mathrm { L } _ { \mathrm { i } } ( \hat { \omega } ) \ell _ { \mathrm { i } } ( \boldsymbol { k } )$ where $k = 2 \pi / \lambda$ is the wavenumber and λ the wavelength, assuming mutually incoherent sources at diferent directions ωˆ. Our goal is to compute the irradiance reaching the sensor due to the scattering of $\operatorname { L } _ { \mathrm { i } } ( \hat { \omega } , \boldsymbol { k } )$ at the surface point $\overrightarrow { p }$

If we zoomed in on the surface around $\overrightarrow { p }$ (Figure 4), we would observe microscopic roughness that we model in Monge form using a random heightfield h. This roughness has scale comparable to the visible wavelength (hence, optically rough) and is not resolvable by the camera, thus the surface appears macroscopically planar. Instead, the roughness is responsible for the surface’s wave-optical scattering behavior, determining its appearance. We model this appearance using the Fourier transform of a functional of the heightfield, as follows.

## Definition 3: Sample BRDF

Given a local surface heightfield $\mathrm { h } ( \cdot )$ , we define the sample bidirectional reflectance distribution function (sample BRDF) $\mathrm { f _ { r } ^ { h } }$ as:

$$
\mathrm { f } _ { \mathrm { r } } ^ { \mathrm { h } } \big ( \vec { q } , \vec { u } , k \big ) : = \frac { 1 } { A } \int _ { \mathbb { R } ^ { 2 } } S \big ( \vec { q } , \vec { r } , k , u _ { z } \big ) e ^ { \mathrm { j } k \vec { u } _ { x y } \cdot 2 \vec { r } } \mathrm { d } \vec { r } ,\tag{4}
$$

where $S$ is a functional of the heightfield h and reflection coeficient ${ \mathrm { r } } ,$

$$
\begin{array} { r l } & { S ( \vec { q } , \vec { r } , k , u _ { z } ) : = \mathrm { s } ( \vec { q } + \vec { r } ; k , u _ { z } ) \mathrm { s } ^ { * } \big ( \vec { q } - \vec { r } ; k , u _ { z } \big ) , } \\ & { \qquad \mathrm { s } \big ( \vec { \rho } ; k , u _ { z } \big ) : = \mathrm { r } \big ( \vec { \rho } , k \big ) e ^ { \mathrm { j } k u _ { z } \mathrm { h } ( \vec { \rho } ) } , \quad \vec { u } \in \mathbb { R } ^ { 3 } . } \end{array}\tag{5}
$$

We elaborate on roughness assumptions and required properties for h in the supplement. We can use $\bar { \mathrm { f _ { r } ^ { h } } }$ , which is random through h, to express the irradiance.<sup>2</sup>

Proposition 4: Appearance model   
The irradiance received at sensor point $\overrightarrow { p }$ equals:   
$\mathrm { E } ( \vec { p } ) \approx \int _ { \mathbb { R } ^ { + } } \int _ { \mathbb { R } ^ { 2 } } \mathrm { w } _ { \mathrm { c } } ( k ) \ell _ { \mathrm { i } } ( k ) \mathrm { P } _ { \mathrm { c } } ( \vec { p } - \vec { q } ) \mathrm { L } _ { \mathrm { o } } ( \vec { q } , \hat { v } , k ) \mathrm { d } \vec { q } \mathrm { d } k ,$ (6)   
outgoing radiance $\mathrm { L } _ { \mathrm { o } }$ at surface point $\overrightarrow { q }$ , direction $\hat { v } ,$ and wavenumber k is:   
\labe { qn:fs\_r ct } di ou p mal , \ ve  w n b r } co q i t \_{ u h m pa \ ld fs ren o c t , i v + u  \wa mb } r d n e p { i v c    \ ot mald r u s g p en { i v c }. fh(q, ω + 0, k)L(ω)( · n) dσ(ω). (7)   
JH2(ñ)

Relationship to classical appearance model. Equation (6) is equivalent to the standard PSF-based image formation model for a thin lens. Equation (7) is almost the standard reflectance equation in computer vision, except it uses the sample BRDF $\mathrm { f _ { r } ^ { h } }$ instead of the standard BRDF $\dot { \overline { { \mathrm { f } } } } _ { \mathrm { r } }$ for rough surfaces as defined in, $e . g .$ , Stam [65]. This diference is the basis for our texture theory: $\mathrm { f _ { r } ^ { h } }$ , and thus also $\mathrm { E } ,$ are random variables due to their dependence on h. Classical appearance models eliminate this randomness by using ensemble averages $\mathbb { E } _ { \mathrm { h } } \left[ \mathrm { f } _ { \mathrm { r } } ^ { \mathrm { h } } \right] , \mathbb { E } _ { \mathrm { h } } [ \mathrm { E } ]$ over all microgeometry realizations. We make the link to classical models explicit.

$$
\mathbb { E } _ { \mathrm { h } } \big [ \mathrm { f } _ { \mathrm { r } } ^ { \mathrm { h } } ( \vec { q } , \hat { \omega } + \hat { v } , k ) \big ] = \overline { { \mathrm { f } } } _ { \mathrm { r } } \big ( \vec { q } , \hat { \omega } + \hat { v } , k \big ) .
$$

From linearity of expectation, replacing $\mathrm { f _ { r } ^ { h } }$ with $\overline { { \mathrm { f } } } _ { \mathrm { r } }$ in Equations (6) and (7) produces the classical radiometric equations for the ensemble-averaged irradiance. Subjective speckle as second-order texture. We use the notation:

$$
\begin{array} { r } { \overline { { \mathrm { E } } } ( \overrightarrow { p } ) : = \mathbb { E } _ { \mathrm { h } } [ \mathrm { E } ( \overrightarrow { p } ) ] , \quad \mathrm { E } ^ { \mathrm { s } } ( \overrightarrow { p } ) : = \mathrm { E } ( \overrightarrow { p } ) - \overline { { \mathrm { E } } } ( \overrightarrow { p } ) . } \end{array}\tag{8}
$$

The diference E<sup>s</sup> between irradiance and its ensemble-average is known in optics as subjective speckle [46, 62]. E<sup>s</sup> arises because of wavelength-scale surface microgeometry that cannot be resolved by the camera at the working magnification. Resolvable changes in surface geometry or material, including visible surface roughness at high magnifications [9], are instead represented in the BRDF $\overline { { \mathrm { f } } } _ { \mathrm { r } }$ (which includes foreshortening), and thus E.

![](images/1cbe0394544c22de8bfbd1b33968995e0ec31e14b5a17ccf18c5332beab51159.jpg)

Returning to DFF, classical computer vision theory defines “texture” as spatial variations of the ensemble-averaged irradiance E (Figure 4). However, the presence of subjective speckle E<sup>s</sup> suggests that even a surface that would classically be considered textureless— $i . e . ,$ one with spatially constant $\mathrm { B R D F - c a n }$

Figure 4: First-order texture (checkerboard pattern) is due to visible BRDF variations. Second-order texture (speckle) is due to invisible random microgeometry even at first-order textureless regions.

show irradiance variations due to non-resolvable microgeometry. We use the terms first-order and second-order texture to distinguish the two types of variations, as one considers only the first moment of E (ensemble average E), whereas the other considers also second moments<sup>3</sup> of E (variance of subjective speckle E<sup>s</sup>).

When is ensemble averaging accurate? The ensemble averaging assumption of classical computer vision is accurate under geometric camera models $( e . g .$ perspective, orthographic) with an infinitesimal aperture and thus infinite PSF in Equation (6). Most previous derivations of appearance models for rough surfaces [64–66] assume an infinite PSF, making them inaccurate for lens-based cameras under large-aperture or close-focus conditions—and thus DFF. Our derivation of Proposition 4 assumes finite PSFs to study implications of deviations from ensemble averaging for DFF.

## 4.2 Characterizing second-order texture contrast

Our next goal is to derive an expression for the coeficient of variation $\kappa _ { \mathrm { E } } ^ { 2 }$ —and thus $\kappa _ { \mathrm { I } } ^ { 2 } .$ —that accounts for both first-order and second-order texture. We can

then use this expression to assess the practicality of second-order texture for DFF, and characterize recoverability improvements.We define

$$
\boldsymbol { A } _ { \mathrm { i } } ( 2 \vec { r } , \boldsymbol { k } ) : = \int _ { \mathcal { H } ^ { 2 } ( \hat { n } ) } \mathrm { L } _ { \mathrm { i } } ( \boldsymbol { \hat { \omega } } ) ( \boldsymbol { \hat { \omega } } \cdot \boldsymbol { \hat { n } } ) e ^ { \mathrm { j } \boldsymbol { k } 2 \vec { r } \cdot \boldsymbol { \hat { \omega } } _ { x y } } \mathrm { d } \sigma ( \boldsymbol { \hat { \omega } } )\tag{9}
$$

as the coherence function (CF) of the far-field illumination $\mathrm { { L } _ { i } } .$ , i.e. its (projected) Fourier transform. The efective support size of $\varLambda _ { \mathrm { i } }$ is the coherence area $\varDelta _ { \mathrm { i } } ^ { 2 }$ [66, 71], which increases as the illumination’s directional footprint decreases Combining Equations (4), (7) and (9) gives $\operatorname { L } _ { \mathrm { o } } ( \vec { q } , \hat { v } , \boldsymbol { k } ) = \mathcal { F } _ { \vec { r } } \{ S \cdot A _ { \mathrm { i } } \} ( \hat { v } )$ , which we use with Proposition 4 to rewrite $\operatorname { E } ( { \overrightarrow { p } } )$ as (denoting $\mathrm { f } ( k ) = \mathrm { w } _ { \mathrm { c } } ( k ) \ell _ { \mathrm { i } } ( k ) )$ :

\my athboxnar ow {intGrayL}{intGrayL }{\vspace {-0.5 em}\c nteri g spectral integra ion ver band \$\bwk \$}{\int \_{\R ^+}\! \s f pectrumincprod \paren {\wavenumb r } \my athboxnar ow {intCr mL}{intCr mL }{\vspace {-0.5 em}\c nteri g spati l ntegra ion ver a \$\dfl ^2\$}{\int \_{\R ^2}\! \psf aren {\pointcam -\pointcam lt } \my athboxnar ow {intBlueL}{intBlueL }{\vspace {-0.5 em}\c nteri g spati l ntegra ion ver a \$\scl ^2\$}{\int \_{\R ^2}\! \pstf \paren {\pointcam lt , \pointcamdif , \wavenumb r } \mcfint \paren {2\pointcamdif , \wavenumb r } e^{\im wavenumb r 2\pointcamdif \cdot \ irvecout \_{xy} \ud { pointcamdif } \ud pointcam lt } \ud wavenumb r }.\vspace {-0.5em} $\varDelta _ { k }$

$$
\begin{array} { c } { { \mathrm { s p a t i a l ~ i n t e g r a t i o n ~ o v e r ~ a r e a ~ \mathcal { A } ^ 2 _ c ~ } } } \\ { { \mathrm { s p a t i a l ~ i n t e g r a t i o n ~ o v e r ~ a r e a ~ \mathcal { A } ^ 2 _ i ~ } } } \\ { { \displaystyle \int _ { \mathbb { R } ^ { + } } \mathrm { f } ( k ) \ \int _ { \mathbb { R } ^ { 2 } } \mathrm { P } _ { \mathrm { c } } ( \overrightarrow { p } - \overrightarrow { q } ) \ \int _ { \mathbb { R } ^ { 2 } } S ( \overrightarrow { q } , \overrightarrow { r } , k ) A _ { \mathrm { i } } ( 2 \overrightarrow { r } , k ) e ^ { \displaystyle \mathrm { j } k 2 \overrightarrow { r } \cdot \hat { v } _ { x y } } \mathrm { d } \overrightarrow { r } \ \mathrm { d } \overrightarrow { q } \ \mathrm { d } k \ . } } \end{array}\tag{10}
$$

We can interpret this expression as follows (Figure 5):

1. The innermost integral equals $\mathrm { L } _ { \mathrm { o } } \big ( \overrightarrow { q } , \hat { v } , \boldsymbol { k } \big )$ , and aggregates complex and random (due to the dependence of S on h) values over points $\vec { r }$ in an area of size $\varDelta _ { \mathrm { i } } ^ { 2 }$ controlled by the CF $\varLambda _ { \mathrm { i } }$ . As the integrand is the product of conjugate-symmetric functions, the integral is real-valued<sup>4</sup> for all $\overrightarrow { q }$

![](images/44770e895b75056ed3e6425afdf54836d94a74b62fcb394355052131a200b3a6.jpg)

2. The intermediate integral equals the spectral density of irradiance $\operatorname { E } ( { \overrightarrow { p } } , k )$ , and aggregates real random $\mathrm { L } _ { \mathrm { o } } \big ( \overrightarrow { q } , \hat { v } , \boldsymbol { k } \big )$ values over points $\overrightarrow { q }$ in an area of focusdependent size $\varDelta _ { \mathrm { c } } ^ { 2 }$ controlled by the PSF $\mathrm { P _ { c } }$

3. The outermost integral aggregates real random $\operatorname { E } ( { \overrightarrow { p } } , k )$ values over wavenumbers k in a spectral range $\varDelta _ { k }$ controlled by the illuminant $\ell _ { \mathrm { i } }$ and SSF $\mathrm { w } _ { \mathrm { c } } .$

Using optics terminology, the innermost integral performs coherent summation within each coherence area—resulting in random $\operatorname { L } _ { \mathrm { o } } ( \vec { q } , \hat { v } , \boldsymbol { k } )$ values manifesting as speckle—

Figure 5: Coherent summation within coherence areas results in speckle (random radiance). Incoherent summation of such areas within the PSF reduces speckle contrast.

whereas the other two perform incoherent summation within the PSF area and spectral bandwidth. Assuming h has suficiently fast variation within each coherence area, the random $\operatorname { L } _ { \mathrm { o } } \big ( \overrightarrow { q } , \hat { v } , k \big )$ values are approximately independent for points $\overrightarrow { q }$ separated by more than $\varDelta _ { \mathrm { i } } \ ( i . e .$ , non-overlapping $\varDelta _ { \mathrm { i } } ^ { 2 } .$ -sized integration areas), or for suficiently separated wavenumbers k—manifesting as uncorrelated speckle. Then, we can understand $\operatorname { E } ( { \overrightarrow { p } } )$ as averaging independent real-valued random radiances whose number scales proportionally to $\varDelta _ { k } \varDelta _ { \mathrm { c } } ^ { 2 } / \varDelta _ { \mathrm { i } } ^ { 2 }$ . Consequently, the variance of $\operatorname { E } ( { \overrightarrow { p } } )$ is inversely proportional to this number, tending to 0 as defocus (and thus $\varDelta _ { \mathrm { c } } ^ { 2 } )$ or bandwidth (and thus $\varDelta _ { k } )$ increase—reducing speckle contrast and converging to ensemble averaging (Figure 6)—while attaining its maximal value when in focus. We formalize this discussion next.

Proposition 6: Pointwise coeficient of variation of irradiance   
In focus, the coeficient of variation of intensity and irradiance equal   
first-order   
\labe { qn:to \_ x ur } cfvi s y = d p m h Bl eL { nt \ ri g s co - } f a 1 u p b k mw d en { + \ i o s c v r a }  y thb x C mL en g fi -o d {\vp rac 1} u s t lb k mw i n e , \v p {- κE .2 MQ(1 + Qk{) k{ (11)   
where: $1 . ~ \kappa _ { \overline { { \mathrm { E } } } } ^ { 2 }$ is the in-focus coeficient of variation for first-order texture E;   
$2 . \ Q : = \pi ^ { \textstyle \Delta _ { \mathrm { c , i f } } ^ { - } } / \Delta _ { \mathrm { i } } ^ { 2 }$ is the relative size of the in-focus PSF and coherence area;   
$3 . \ M : = G _ { 1 } \varDelta _ { k }$ is proportional to the spectral bandwidth, through a constant   
$G _ { 1 }$ that depends on statistical properties of h and the mean wavevector.

Notably, we consider a surface without first-order texture, meaning constant ensemble-averaged irradiance ${ \overline { { \mathrm { E } } } } ,$ and thus $\kappa _ { \overline { { \mathrm { E } } } } ^ { 2 } = 0$ . Such a surface would be classically described as “textureless.” Yet, Equation (11) predicts non-zero in-focus texture contrast $\kappa _ { \mathrm { I } } ^ { 2 } = 1 / M Q$ , due to second-order texture. Therefore, from Proposition 2, even this first-order textureless surface is potentially recoverable, so long as the contrast is stronger than noise. Equation (11) further predicts that we can increase this contrast by reducing the bandwidth $\varDelta _ { k }$ and thus $\scriptstyle { M - e . g . }$ , by using a spectral filter.<sup>5</sup> Assessing recoverability requires considering the impact on SNR, as we do in $\ S ~ 5$

<table><tr><td>450 nm</td><td>532 nm</td></tr><tr><td>560 nm</td><td>no filter</td></tr></table>

Figure 6: In-focus images of a white target using diferent filters (center λ) show uncorrelated speckle patterns. Their incoherent summation without a filter eliminates speckle.

Practicality for computer vision. The factor Q in Equation (11) helps understand under what conditions second-order texture is practical for DFF.

Its in-focus contrast improves with larger coherence areas $\varDelta _ { \mathrm { i } } ^ { 2 }$ due to illumination dominated by concentrated far-field sources $( e . g .$ , outdoor sunlight, indoor small ceiling lights), and worsens with more directionally uniform illumination $( e . g .$ outdoor overcast conditions, indoor large ceiling panels). It also improves with smaller in-focus PSF sizes $\varDelta _ { \mathrm { c , i f } } ^ { 2 } , i . e .$ , smaller f-numbers or reproduction ratios [74]. In $\ S \ G$ we show that, using bandwidths $\varDelta _ { \lambda } = 1 0 \mathrm { - } 1 0 0 \mathrm { n m }$ , second-order texture improves $\mathrm { D F F }$ in settings typical for passive computer vision: 1. both outdoor and indoor ambient lighting $( \varDelta _ { \mathrm { i } } \approx 5 0 \mathrm { - 1 0 0 \mu m \ [ 4 5 ] ) ; 2 }$ . reproduction ratios $^ 1 / 5 0 ^ { - 1 } / 1 0$ at f-numbers up to 6 $( \varDelta _ { \mathrm { c , i f } } \approx 3 0 \mathrm { - } 1 5 0 \mu \mathrm { m } )$ . Second-order contrast reduces also with

larger pixel sizes [44, 75], favoring smaller pixel pitches (3 µm in our experiments).   
We elaborate in the supplement.

## 5 Recoverability with second-order texture

We focus on recoverability for a first-order textureless surface, i.e., $\kappa _ { \overline { { \mathrm { E } } } } ^ { 2 } = 0 .$ . From Proposition 2, it sufices to express the texture contrast–SNR product $\Theta ^ { 2 }$ as a function of bandwidth $\varDelta _ { k }$ and exposure time t. Equation (11) already provides an expression for $\kappa _ { \mathrm { I } } ^ { 2 } { : }$ ; whereas from Equation (2), getting an expression for $\kappa _ { n } ^ { 2 }$ requires relating the mean flux $\mu _ { \Phi }$ to $\varDelta _ { k }$ . We prove the following.

Proposition 7: Texture contrast–SNR product   
In focus, the texture contrast–SNR product $\Theta ^ { 2 } : = \left. \kappa _ { \mathrm { I } } ^ { 2 } \right/ \kappa _ { n } ^ { 2 }$ equals:   
$\Theta ^ { 2 } ( t , \Delta _ { k } ) \approx \frac { 1 } { Q } \frac { G _ { 2 } } { G _ { 1 } } \frac { t } { \left( 1 + \frac { \sigma _ { \mathrm { r e a d } } ^ { 2 } } { G _ { 2 } } \frac { 1 } { \Delta _ { k } t } \right) } ,$ (12)   
where the exposure time $t \in [ 0 , g \mathrm { I } _ { \operatorname* { m a x } } / G _ { 2 } \varDelta _ { k } )$ has a saturation limit dependent on   
spectral bandwidth $\varDelta _ { k }$ , and the constant $G _ { 2 }$ depends on spectral reflectance.

Analysis and simulation. In Figure 7, we visualize the probability of error $\mathrm { P r } ( \widetilde { \mathrm { c } } _ { \mathrm { i f } } ^ { 2 } < \widetilde { \mathrm { c } } _ { \mathrm { o o f } } ^ { 2 } )$ predicted from Equations (3) and (12) for diferent bandwidths $\varDelta _ { k }$ e eand exposure times t. We also use Monte Carlo simulation to numerically estimate $\mathrm { P r } ( \widetilde { \mathrm { c } } _ { \mathrm { i f } } ^ { 2 } < \widetilde { \mathrm { c } } _ { \mathrm { o o f } } ^ { 2 } )$ and assess the accuracy of e eour theory. Assuming a first-order textureless surface, we sampled focal stacks of noisy speckle patterns, then computed sample contrast $\mathrm { { \widetilde { c } } ^ { 2 } }$ . We sampled speckle eusing a statistical model detailed in the supplement, and sensor noise using the model of § 3. The supplement lists all parameters (illumination, surface, sensor, lens) and other details of the visualization and simulation. The results show closely matching theory and simulation predictions for predictions for $\mathrm { P r } ( \widetilde { \mathrm { c } } _ { \mathrm { i f } } ^ { 2 } < \widetilde { \mathrm { c } } _ { \mathrm { o o f } } ^ { 2 } )$ (lower is ebetter). We also observe: better). We also observe:

![](images/d56bc9e9f4d53423f07f998ef4c8865e93589773186c6f838581d855aa6fac76.jpg)

Figure 7: Error probability(lower is $\mathrm { P r } ( \widetilde { \mathrm { c } } _ { \mathrm { i f } } ^ { 2 } < \widetilde { \mathrm { c } } _ { \mathrm { o o f } } ^ { 2 } )$ estimation from theory and simulation.

1. For constant bandwidth (a vertical line), the optimal t is the saturation limit. As texture contrast $\kappa _ { \mathrm { I } } ^ { 2 }$ is fixed, maximizing t maximizes SNR $1 / \kappa _ { n } ^ { 2 }$ and $\varTheta ^ { 2 }$

2. For constant exposure time (a horizontal line), the optimal $\varDelta _ { k }$ is again determined by the saturation limit, i.e., maximizing $\varDelta _ { k }$ till intensities saturate. The decrease in $\kappa _ { \mathrm { I } } ^ { 2 }$ is outweighed by the increase in $1 / \kappa _ { n } ^ { 2 }$ , improving $\theta ^ { 2 }$

3. For constant bandwidth–exposure time product (a diagonal line, corresponding to constant SNR), optimizing $\Theta ^ { 2 }$ requires minimizing $\varDelta _ { k }$ while proportionally increasing $t ,$ to stay close to the saturation limit.

![](images/acbbde1807b7dbd8d81c3abf401a13021656b9a842f0b657f608665cdfb1aa0a.jpg)

![](images/ee012ad6dcc167e2dba60dc6b95b61326a96c3454e6ba229c2dfab2160dfee9d.jpg)

![](images/dbe6f3345e0e8bf975eebae5d2a9f08928b9298cc4eda54ec3aeab0eca79fb76.jpg)

![](images/598f01d8778f1bd30495660e7c923cfba29c7d35b140006b0a6a2cc3ee8bd3d1.jpg)

![](images/a79de6f5e9c2be193fb8489de2a1ebe8868cb7edd587450d08b2bc5a59141635.jpg)  
Figure 8: DFF improves with filters of decreasing bandwidth thanks to enhanced second-order texture, visible in median-subtracted crops (‘med-sub’). This enhancement also results in focus measure peaks more distinguishable from noise (top two plots), decreasing the fraction of pixels under a z-score threshold (bottom two plots, gray pixels in depth maps). Example captured outdoors under sunlight.

4. When decreasing $\varDelta _ { k }$ , it is also possible to improve $\theta ^ { 2 }$ by increasing t sublinearly, i.e., by less than the amount required to maintain constant SNR.

5. Due to the saturation limit for a single capture (diagonal discontinuity in simulated results), lowering $\varDelta _ { k }$ with a filter enables much reduced probability of error (top–left corner) than without a filter (bottom–right corner).

In the next section, we validate these observations through extensive experiments.

## 6 Experiments & limitations

We perform DFF experiments across multiple scenes using only pre-existing ambient light—either sunlight, or standard ceiling lights. The supplemental PDF and project website include more experiments, visualizations, code, and data.

Experimental setup. Our setup uses a machine vision camera (FLIR BFS-U3-122S6M-C), a motorized lens mount for focus control, a photographic lens (Canon EF-S 60mm f/2.8 Macro USM), and narrowband spectral filters (Edmund Optics, diferent FWHM bandwidths, center wavelength  530 nm).

![](images/be81160fbc05b024e2a3997344d16f0d3a181521f19c5de6613f64117f39b9aa.jpg)

Implementation details. Given a focal stack $\{ \mathrm { I } _ { j } ( \vec { p } ) \} _ { j = 1 } ^ { J } { } ;$ , we use a focus measure that aggregates $3 \times 3$ Laplacian values over a Gaussian kernel of width $\sigma _ { \mathrm { a g g } } = 5 / 2$ [68]. Empirically, this measure performed better than the sample contrast. For simplicity, we continue to use ${ \widetilde { \mathrm { c } } } ^ { 2 }$ for this measure.

![](images/46bcff4d17bdf5b04558c05146c646f971ed4555fbba105bbba174dbd1357062.jpg)

eAs we do not have access to ground-truth depth in general, we quantify DFF performance using the per-pixel robust z-score $\xi : = | \widetilde { \mathrm { c } } _ { \ast } ^ { 2 } \mathrm { - m e d i a n } ( \widetilde { \mathrm { c } } _ { j } ^ { 2 } ) | / \mathrm { M A D } ( \widetilde { \mathrm { c } } _ { j } ^ { 2 } )$ where: 1. $\widetilde { \mathrm { c } } _ { \ast } ^ { 2 }$ is the max focus measure at

Figure 9: Validation of the η metric with active DFF.

$$
j ^ { \star } ; ~ 2
$$

$$
\textstyle ( { \widetilde { \mathrm { c } } } _ { j } ^ { 2 } )
$$

10 nm, well exposed

![](images/7facdf5d39564caabd63ed1c2a88b936611aef58c8f2e97a1cfbefb4c14f2fe5.jpg)

![](images/46103ff23c694a5842a79eef4ecf2d9a92b5fbdb59a14875e72e45bda088f7c8.jpg)

![](images/5d892566ad07d8902a5fed50d705edd600feac9e83c726b384a81d0232a69812.jpg)

![](images/ac8f66d0737a094e75f2118eba70bd485b7a3d9ec056a076ba30800dcb92cd1a.jpg)

![](images/1600bf1760e87874e2aca23c1b04d802e96dbfe7bcecc135892bae159dd08525.jpg)

![](images/8320fe625df4e60bb0dd3e6d0fc223bf9bfa9cf41535e95328566241a91227e9.jpg)

![](images/945709b70ad038de8b9c75777b533d61ff88fa0571ee8a72b89221e2f4e05e5b.jpg)

![](images/1cb08572f79cb1b49ad540be984470ca65cb01acec86c4c4d708aa31df3518aa.jpg)  
Figure 10: Using a spectral filter improves DFF even at suboptimal exposures. At $^ 1 / 4$ the optimal exposure time, second-order texture is already discernible from noise at median-subtracted crops (‘ms-if’ vs ‘ms-oof’), improving focus measures and z-score statistics (bottom plots). Example captured indoors under ceiling lights.

$\mathrm { M A D } _ { j } ( \widetilde { \mathrm { c } } _ { j } ^ { 2 } ) : = \mathrm { m e d i a n } _ { j } | \widetilde { \mathrm { c } } _ { j } ^ { 2 } - \mathrm { m e d i a n } ( \widetilde { \mathrm { c } } _ { j } ^ { 2 } ) |$ are the median and median absolute e e edeviation (MAD) of the focus measure curve. Thus, ξ quantifies confidence that the focus measure peak corresponds to the true depth.

We use $\xi$ to filter out noisy depth estimates by thresholding at some level $\xi _ { \mathrm { t h } }$ (shown as gray pixels in depth maps), and to compute an error metric $\eta \in [ 0 , 1 ]$ equal to the fraction $o f$ unrecovered pixels (pixels with $\xi ( \vec { p } ) < \xi _ { \mathrm { t h } } )$ . We use η as a metric to compare DFF performance, as a proxy of $\mathrm { P r } ( \widetilde { \mathrm { c } } _ { \mathrm { i f } } ^ { 2 } < \widetilde { \mathrm { c } } _ { \mathrm { o o f } } ^ { 2 } )$ . In Figure 9, e ewe validate this metric by comparing it to the root-mean-square error (RMSE) with respect to reference depth obtained with active DFF (i.e., using a projector to create artificial texture). The figure shows that RMSE and η are strongly correlated as expected—wrong depth estimates arise from flat focus curves (low texture or high noise), hence have low $\xi$ values. We elaborate in the supplement. Contrast enhancement. Figure 8 shows enhanced in-focus texture and drastically improved depth using filters of decreasing bandwidth, while adjusting exposure time to maintain exposure. Median-subtracted images of a weakly

textured patch help visualize the contrast enhancement. The plots show focus curves and z-score statistics. Figures 1, 2 and 13 show more scenes.

![](images/03c18c77091c42d0b2af0aad2b76cd9f0888fb4de49953ef0c664dfc9a6f4845.jpg)

![](images/88e3b73980b1041c1d5565c45652830e4de7ef17e0c247f189e8a89a2bfb4ce4.jpg)

Exposure impact. Figures 10 and 11   
assess the tradeof between exposure   
Figure 11: DFF improves with a spectraltime and spectral bandwidth. In Fig- time and spectral bandwidth. In Fig  
ure 10, adding a filter of bandwidth filter even at suboptimal exposures. Exam-  
$\varDelta _ { \lambda } = 1 0$ nm requires increasing expo-

ple captured outdoors under sunlight.

![](images/a9dbff567c50190bc08d44017e9c9b4ec8b941d761ac97d36ef1de4483433a1e.jpg)  
Figure 12: Aggregating focus measures over progressively larger patch sizes allows depth recovery at adaptive scales. With the filter, much finer details are recovered at smaller patch sizes (see resolution map). Example captured outdoors under sunlight.

sure time by $3 2 \times$ to maintain proper exposure. Yet DFF performance already improves when we increase exposure time by just 4 (i.e., underexposure by 3 stops), with further improvements closer to proper exposure. The crops and plots show the increase in contrast and z-scores, at a weakly textured patch—where second-order texture gradually overcomes noise—and in aggregate. The same behavior holds for other bandwidths (bottom right) and outdoors (Figure 11).

Pixel-adaptive aggregation. So far we considered DFF with a fixed focus measure aggregation width $\sigma _ { \mathrm { a g g } }$ (or efectively, fixed patch size T) at all pixels. However, the optimal aggregation width can vary per pixel, depending on local texture contrast and noise level. We use a simple adaptive method that, at each pixel, selects the minimum aggregation width $\sigma _ { \mathrm { a g g } } ^ { * }$ that achieves z-score above a threshold $\xi _ { \mathrm { t h } }$ . We can then use $\sigma _ { \mathrm { a g g } } ^ { * }$ as a metric to assess improvements in lateral resolution due to second-order texture. Figure 12 shows that using a spectral filter indeed results in depth maps of significantly higher lateral resolution.

Limitations. The supplement shows experiments representative of the two main cases where using a spectral filter may not improve DFF performance: The first is translucent surfaces, as subsurface scattering greatly diminishes speckle contrast [46, §6.4.3]. The second is illumination with large directional footprint $( e . g .$ ， indoor lights that are very large or very close to the scene, overcast outdoor conditions, or strong indirect light), making $\varDelta _ { \mathrm { i } }$ much smaller than $\varDelta _ { \mathrm { c , i f } }$ . We show that the deterioration of contrast, and thus DFF performance, with increasing directional bandwidth is consistent with our theory (§ 4.2).

Moreover, improving DFF performance with a spectral filter requires increasing exposure time—even if sublinearly. Figures 7 and 10 show that a filter bandwidth $\varDelta _ { \lambda } = 1 0$ nm requires a 4 exposure time increase before performance starts to improve. Our results should help assess whether diferent application settings justify this tradeof between depth accuracy and acquisition time.

## 7 Conclusion

We developed a second-order theory of texture, with significant implications for the definition of texture and its role in passive depth from focus. Our theory shows that: 1. Even surfaces that would traditionally be considered textureless can produce textured appearance, thanks to subjective speckle due to surface microgeometry. 2. Using a narrowband spectral filter enhances the in-focus contrast of this texture. 3. This enhancement can bring about dramatic improvements in depth accuracy under conditions typical for passive computer vision. We validated this theory through extensive simulations and experiments.

![](images/ac2d57bc6b165fd4c72dd0b545442c115856a97428cf996046c83a38d5bd02bd.jpg)  
Figure 13: Examples of DFF improvement in indoor and outdoor real-world scenes. Using a spectral filter improves both fixed-size and pixel-adaptive reconstructions. The project website provides interactive visualizations of these and other scenes.

Our findings invite a broader reexamination of texture and passive depth sensing in computer vision: Though we developed our theory in the context of depth from focus, second-order texture can likely benefit other passive methods using focus or parallax. Further development of algorithms (confocal constancy [17] and focal flow [18] of subjective speckle) and theory (multi-view correlations of subjective speckle [76] and memory efect [57]) can realize these benefits. Additionally, the diferent statistics of speckle and sensor noise hint at the possibility of using learning-based methods to detect speckle even under very low SNR. Future research can adapt related successful approaches in other speckle imaging applications [55, 56]. Lastly, both our work and recent work on passive interferometric depth sensing [43–45] rely on the weak coherence of ambient light to relax texture requirements. Further research should help understand the relative merits of interferometric and non-interferometric methods.

## Acknowledgements

We thank Dorian Chan, Aswin Sankaranarayanan, and Matthew O’Toole for helpful discussions about speckle, and Mian Wei for feedback on writing. This work was supported by NSF award 2047341, and a Sloan Research Fellowship.

## References

1. Sundaram, H., Nayar, S.: Are textureless scenes recoverable? In: IEEE Conf. Comput. Vis. Pattern Recog., pp. 814–820, IEEE (1997) 1

2. Barnard, S.T., Thompson, W.B.: Disparity analysis of images. IEEE Trans. Pattern Anal. Mach. Intell. 2(4), 333–340 (1980) 3

3. Nalpantidis, L., Sirakoulis, G., Gasteratos, A.: Rev. of stereo vision algorithms: Software to hardware. Int. J. Optomechatron. 2(4), 435–462 (2008)

4. Hartley, R., Zisserman, A.: Multiple View Geometry in Computer Vision. Cambridge University Press (2004)

5. Scharstein, D., Szeliski, R.: A taxonomy and evaluation of dense two-frame stereo correspondence algorithms. Int. J. Comput. Vis. 47(1), 7–42 (2002) 3

6. Suwajanakorn, S., Hernandez, C., Seitz, S.M.: Depth from focus with your mobile phone. In: IEEE Conf. Comput. Vis. Pattern Recog., pp. 3497–3506 (2015) 3

7. Hazirbaş, C., Soyer, S.G., Staab, M.C., Leal-Taixé, L., Cremers, D.: Deep depth from focus. In: Asian Conf. Comput. Vis., pp. 525–541 (2018)

8. Grossmann, P.: Depth from focus. Pattern Recognit. Lett. 5(1), 63–69 (1987)

9. Nayar, S.K., Nakagawa, Y.: Shape from focus. IEEE Trans. Pattern Anal. Mach. Intell. 16(8), 824–831 (1994) 4, 8, 11, 12

10. Nayar, S.K., Watanabe, M., Noguchi, M.: Real-time focus range sensor. IEEE Trans. Pattern Anal. Mach. Intell. 18(12), 1186–1198 (1996)

11. Subbarao, M., Choi, T.: Accurate recovery of three-dimensional shape from image focus. IEEE Trans. Pattern Anal. Mach. Intell. 17(3), 266–274 (1995) 3

12. Subbarao, M., Surya, G.: Depth from defocus: A spatial domain approach. Int. J. Comput. Vis. 13(3), 271–294 (1994) 3

13. Favaro, P.: Recovering thin structures via nonlocal-means regularization with application to depth from defocus. In: IEEE Conf. Comput. Vis. Pattern Recog., pp. 1133–1140 (2010)

14. Pentland, A.P.: A new sense for depth of field. IEEE Trans. Pattern Anal. Mach. Intell. 9(4), 523–531 (1987)

15. Tang, H., Cohen, S., Price, B., Schiller, S., Kutulakos, K.N.: Depth from defocus in the wild. In: IEEE Conf. Comput. Vis. Pattern Recog., pp. 2740– 2748 (2017)

16. Watanabe, M., Nayar, S.K.: Rational filters for passive depth from defocus. Int. J. Comput. Vis. 27(3), 203–225 (1998) 3

17. Hasinof, S.W., Kutulakos, K.N.: Confocal stereo. Int. J. Comput. Vis. 81, 82–104 (2009) 3, 15

18. Alexander, E., Guo, Q., Koppal, S., Gortler, S., Zickler, T.: Focal flow: Measuring distance and velocity with defocus and diferential motion. In: Eur. Conf. Comput. Vis., pp. 667–682, Springer (2016) 3, 15

19. Guo, Q., Alexander, E., Zickler, T.: Focal track: Depth and accommodation with oscillating lens deformation. In: Int. Conf. Comput. Vis., pp. 966–974 (2017) 3

20. Levin, A., Fergus, R., Durand, F., Freeman, W.T.: Image and depth from a conventional camera with a coded aperture. ACM Trans. Graph. 26(3), 70 (2007) 3

21. Veeraraghavan, A., Raskar, R., Agrawal, A., Mohan, A., Tumblin, J.: Dappled photography: Mask enhanced cameras for heterodyned light fields and coded aperture refocusing. ACM Trans. Graph. 26(3), 69 (2007)

22. Zhou, C., Nayar, S.K.: What are good apertures for defocus deblurring? In: IEEE Int. Conf. Comput. Photography, pp. 1–8 (2009)

23. Zhou, C., Lin, S., Nayar, S.K.: Coded aperture pairs for depth from defocus. In: Int. Conf. Comput. Vis., pp. 325–332 (2009)

24. Chakrabarti, A., Zickler, T.: Depth and deblurring from a spectrally-varying depth-of-field. In: Eur. Conf. Comput. Vis., pp. 648–661, Springer (2012) 3

25. Schechner, Y.Y., Kiryati, N.: Depth from defocus vs. stereo: How diferent really are they? Int. J. Comput. Vis. 39, 141–162 (2000) 3

26. Bolles, R.C., Baker, H.H., Marimont, D.H.: Epipolar-plane image analysis: An approach to determining structure from motion. Int. J. Comput. Vis. 1(1), 7–55 (1987) 3

27. Kim, C., Zimmer, H., Pritch, Y., Sorkine-Hornung, A., Gross, M.H.: Scene reconstruction from high spatio-angular resolution light fields. ACM Trans. Graph. 32(4), 73:1–73:12 (2013) 3

28. Punnappurath, A., Abuolaim, A., Afifi, M., Brown, M.S.: Modeling defocusdisparity in dual-pixel sensors. In: IEEE Int. Conf. Comput. Photography, pp. 1–12 (2020) 3

29. Punnappurath, A., Brown, M.S.: Reflection removal using a dual-pixel sensor. In: IEEE Conf. Comput. Vis. Pattern Recog., pp. 1556–1565 (2019)

30. Garg, R., Wadhwa, N., Ansari, S., Barron, J.T.: Learning single camera depth estimation using dual-pixels. In: Int. Conf. Comput. Vis., pp. 7628– 7637 (2019)

31. Xin, S., Wadhwa, N., Xue, T., Barron, J.T., Srinivasan, P.P., Chen, J., Gkioulekas, I., Garg, R.: Defocus map estimation and deblurring from a single dual-pixel image. In: Int. Conf. Comput. Vis., pp. 2228–2238 (2021) 3

32. Rosen, J., Vijayakumar, A., Kumar, M., Rai, M.R., Kelner, R., Kashter, Y., Bulbul, A., Mukherjee, S.: Recent advances in self-interference incoherent digital holography. Adv. Opt. Photonics 11(1), 1–66 (2019) 3

33. Liu, J.P., Tahara, T., Hayasaki, Y., Poon, T.C.: Incoherent digital holography: a review. Appl. Sci. 8(1), 143 (2018)

34. Tahara, T., Zhang, Y., Rosen, J., Anand, V., Cao, L., Wu, J., Koujin, T., Matsuda, A., Ishii, A., Kozawa, Y., Okamoto, R., Oi, R., Nobukawa, T., Choi, K., Imbe, M., Poon, T.C.: Roadmap of incoherent digital holography. Appl. Phys. B 128(11), 193 (2022)

35. Tahara, T., Kanno, T., Arai, Y., Ozawa, T.: Single-shot phase-shifting incoherent digital holography. J. Opt. 19(6), 065705 (2017)

36. Tahara, T., Kozawa, Y., Ishii, A., Okamoto, R.: Palm-sized single-shot phaseshifting incoherent digital holography system with birefringent materials. In: Laser Science, pp. JTu5B–52, Optica Publishing Group (2022)

37. Muroi, T., Nobukawa, T., Katano, Y., Hagiwara, K.: Capturing videos at 60 frames per second using incoherent digital holography. Opt. Contin. 2(11), 2409–2420 (2023) 3

38. Davy, M., Fink, M., de Rosny, J.: Green’s function retrieval and passive imaging from correlations of wideband thermal radiations. Phys. Rev. Lett. 110, 203901 (May 2013) 3

39. Badon, A., Lerosey, G., Boccara, A.C., Fink, M., Aubry, A.: Retrieving timedependent Green’s functions in optics with low-coherence interferometry. Phys. Rev. Lett. 114, 023901 (Jan 2015)

40. Badon, A., Li, D., Lerosey, G., Boccara, A.C., Fink, M., Aubry, A.: Spatio temporal imaging of light transport in highly scattering media under white light illumination. Optica 3(11), 1160–1166 (Nov 2016)

41. Boger-Lombard, J., Katz, O.: Passive optical time-of-flight for non line-ofsight localization. Nat. Commun. 10(1), 3343 (Jul 2019)

42. Batarseh, M., Sukhov, S., Shen, Z., Gemar, H., Rezvani, R., Dogariu, A.: Passive sensing around the corner using spatial coherence. Nat. Commun. 9(1), 3629 (Sep 2018) 3

43. Cossairt, O., Matsuda, N., Gupta, M.: Digital refocusing with incoherent holography. In: IEEE Int. Conf. Comput. Photography, pp. 1–9, IEEE (2014) 3, 15

44. Chen, W.Y., Sankaranarayanan, A.C., Levin, A., O’Toole, M.: Coherence as texture—passive textureless 3D reconstruction by self-interference. In: IEEE Conf. Comput. Vis. Pattern Recog., pp. 25058–25066 (2024) 3, 11

45. Kotwal, A., Levin, A., Gkioulekas, I.: Passive micron-scale time-of-flight with sunlight interferometry. In: IEEE Conf. Comput. Vis. Pattern Recog., pp. 4139–4149 (2023) 3, 10, 15

46. Goodman, J.W.: Speckle phenomena in optics: theory and applications. SPIE (2020) 3, 6, 8, 10, 14, 5, 16, 19, 23, 25, 27

47. Mertz, J.: Introduction to optical microscopy. Cambridge University Press (2019) 2

48. Akkermans, E., Montambaux, G.: Mesoscopic physics of electrons and photons. Cambridge university press (2007) 3

49. Shih, Y.C., Davis, A., Hasinof, S.W., Durand, F., Freeman, W.T.: Laser speckle photography for surface tampering detection. In: IEEE Conf. Comput. Vis. Pattern Recog., pp. 33–40, IEEE (2012) 3

50. Jo, K., Gupta, M., Nayar, S.K.: Spedo: 6 dof ego-motion sensor using speckle defocus imaging. In: Int. Conf. Comput. Vis., pp. 4319–4327 (2015) 3

51. Smith, B.M., Desai, P., Agarwal, V., Gupta, M.: CoLux: Multi-object 3D micro-motion analysis using speckle imaging. ACM Trans. Graph. 36(4), 1–12 (2017)

52. Smith, B.M., O’Toole, M., Gupta, M.: Tracking multiple objects outside the line of sight using speckle imaging. In: IEEE Conf. Comput. Vis. Pattern Recog., pp. 6258–6266 (2018) 3

53. Kichler, M., Bagon, S., Sheinin, M.: Learning to see inside opaque liquid containers using speckle vibrometry. In: Int. Conf. Comput. Vis., pp. 9466– 9476 (2025) 3

54. Sheinin, M., Chan, D., O’Toole, M., Narasimhan, S.G.: Dual-shutter optical vibration sensing. In: IEEE Conf. Comput. Vis. Pattern Recog., pp. 16324– 16333 (2022) 3

55. Metzler, C.A., Heide, F., Rangarajan, P., Balaji, M.M., Viswanath, A., Veeraraghavan, A., Baraniuk, R.G.: Deep-inverse correlography: towards real-time high-resolution non-line-of-sight imaging. Optica 7(1), 63–71 (2020) 3, 15

56. Xie, M., Guo, H., Feng, B.Y., Jin, L., Veeraraghavan, A., Metzler, C.A.: WaveMo: learning wavefront modulations to see through scattering. In: IEEE Conf. Comput. Vis. Pattern Recog., pp. 25276–25285 (2024) 3, 15

57. Alterman, M., Bar, C., Gkioulekas, I., Levin, A.: Imaging with local speckle intensity correlations: theory and practice. ACM Trans. Graph. 40(3), 1–22 (2021) 15

58. Boniface, A., Blochet, B., Dong, J., Gigan, S.: Noninvasive light focusing in scattering media using speckle variance optimization. Optica 6(11), 1381–1385 (2019) 3

59. Kim, J., Benko, C., Wrenninge, M., Villemin, R., Barber, Z., Jarosz, W., Pediredla, A.: A monte carlo rendering framework for simulating optical heterodyne detection. ACM Trans. Graph. 44(4), 1–19 (2025) 3

60. Bar, C., Gkioulekas, I., Levin, A.: Rendering near-field speckle statistics in scattering media. ACM Trans. Graph. 39(6), 1–18 (2020)

61. Bar, C., Alterman, M., Gkioulekas, I., Levin, A.: A monte carlo framework for rendering speckle statistics in scattering media. ACM Trans. Graph. 38(4), 1–22 (2019)

62. Steinberg, S., Yan, L.Q.: Rendering of subjective speckle formed by rough statistical surfaces. ACM Trans. Graph. 41(1), 2:1–2:23 (Feb 2022) 8, 16

63. Liu, Z., Huo, Y., Peng, Y., Wang, R.: A fully-statistical wave scattering model for heterogeneous surfaces. ACM Trans. Graph. 44(4), 1–17 (2025) 3

64. Beckmann, P., Spizzichino, A.: The scattering of electromagnetic waves from rough surfaces. Artech House (1987) 3, 6, 8, 19

65. Stam, J.: Difraction shaders. In: ACM SIGGRAPH, pp. 101–110 (1999) 7, 20, 22

66. Levin, A., Glasner, D., Xiong, Y., Durand, F., Freeman, W., Matusik, W., Zickler, T.: Fabricating BRDFs at high spatial resolution using wave optics. ACM Trans. Graph. 32(4), 1–14 (2013) 6, 8, 9, 19, 20

67. Dong, Z., Walter, B., Marschner, S., Greenberg, D.P.: Predicting appearance from measured microgeometry of metal surfaces. ACM Trans. Graph. 35(1), 1–13 (2015) 3

68. Subbarao, M., Tyan, J.K.: Selecting the optimal focus measure for autofocusing and depth-from-focus. IEEE Trans. Pattern Anal. Mach. Intell. 20(8), 864–870 (1998) 4, 12, 11

69. Hasinof, S.W., Durand, F., Freeman, W.T.: Noise-optimal capture for high dynamic range photography. In: IEEE Conf. Comput. Vis. Pattern Recog., pp. 553–560, IEEE (2010) 5

70. Cuypers, T., Haber, T., Bekaert, P., Oh, S.B., Raskar, R.: Reflectance model for difraction. ACM Trans. Graph. 31(5), 1–11 (2012) 6

71. Wolf, E.: Introduction to the Theory of Coherence and Polarization of Light. Cambridge university press (2007) 9, 16

72. Testorf, M.E., Hennelly, B.M., Ojeda-Castañeda, J.: Phase-space optics: fundamentals and applications. The McGraw-Hill Companies, Inc. (2010) 9, 17

73. Steinberg, S., Pharr, M.: Wave tracing: Generalizing the path integral to wave optics. Comput. Graph. Forum p. e70322 (2026) 9, 19

74. Kingslake, R.: The efective aperture of a photographic objective. J. Opt. Soc. Am. 35(8), 518–520 (1945) 10, 2

75. Gkioulekas, I., Levin, A., Durand, F., Zickler, T.: Micron-scale light transport decomposition using interferometry. ACM Trans. Graph. 34(4), 37:1–37:14 (Jul 2015) 11

76. Fienup, J.R., Idell, P.S.: Imaging correlography with sparse arrays of detectors. Opt. Eng. 27(9), 778–784 (1988) 15

# A second-order theory of texture for depth from focus — Supplemental Document —

project webpage: https://imaging.cs.cmu.edu/second-order-texture

This document is broadly organized into two parts. §§ A–D are focused on additional results, analyses and implementation details, while §§ E–I are dedicated to theoretical derivations including proofs of all of the main results.

## Table of Contents

A Practical conditions for second-order texture 2   
A.1 Imaging: lens settings and pixel pitch .   
A.2 Illumination conditions . . .   
A.3 Material constraints .   
B Interplay between texture contrast and noise   
C Depth recovery performance evaluation 9   
C.1 Quantitative comparison . 9   
C.2 Validation of z-score-based metric with active DFF reference 11   
D Implementation details 11   
D.1 Capture setup 11   
D.2 Depth recovery algorithms . 12   
D.3 Monte Carlo simulation 12   
E Appearance model for second-order texture . . 16   
E.1 Incident illumination model . 16   
E.2 Coherence function of the incident illumination . . 18   
E.3 Scattering and Measurement . 19   
E.4 Approximating measured irradiance . . 20   
E.5 Ensemble averaging . . 22   
F Characterization of second-order texture . 24   
F.1 Proof of exponential distribution 27   
F.2 Spatial integration . . 28   
G Noise model . 30   
H Depth from focus analysis . . 31   
H.1 Statistics of the focus measure 32   
H.2 Proof of Proposition 2 34   
Recoverability with second-order texture . 36   
I.1 Attenuation due to spectral filtering 37   
I.2 Gaussian spectral profile . 37   
I.3 Wavenumber and wavelength bandwidths . 38

## A Practical conditions for second-order texture

The in-focus contrast of second-order texture, and hence DFF performance, depends on practical factors spanning the imaging system, the illumination, and the surface material. The analysis and experiments presented in this section serve two purposes: 1. they validate our theoretical predictions on the nature of second-order texture in § 4.2; and 2. they demonstrate limitations of our technique, i.e. regimes where using a narrowband filter ofers little to no benefit.

## A.1 Imaging: lens settings and pixel pitch

In § 4.2, we showed that the in-focus contrast of second-order texture $\kappa _ { \mathrm { I } } ^ { 2 }$ is dependent on the in-focus PSF width $\varDelta _ { \mathrm { c , i f } }$ via Q–the relative size of the in-focus PSF and the coherence area. We characterize the impact of practical camera parameters—lens f-number and reproduction ratio (that together determine PSF width), and sensor pixel pitch—on the in-focus second-order texture contrast and DFF performance, and derive optimal parameter values. We excluded the impact of sensor pixel pitch in our analysis in § 4 for the sake of analytical tractability (by assuming flux is proportional to irradiance, and hence that $\kappa _ { \mathrm { I } } ^ { 2 } = \kappa _ { \mathrm { E } } ^ { 2 } ) -$ the role of which will become evident below.

Theoretical analysis. We can use Equation (11) to determine how the lens’ f-number N and reproduction ratio m change second-order texture contrast, through their impact on the lens’ in-focus PSF size $\varDelta _ { \mathrm { c , i f } }$ . At infinity focus, the image-space width of the difraction-limited PSF equals λN [47, §5.2.2]. We can account for finite focus with reproduction ratio m by replacing f-number N with the efective f-number $N _ { \mathrm { e f f } } : = ( m + 1 ) N . [ 7 4 ]$ Lastly, we can convert to object-space widths by dividing by the reproduction ratio m, arriving at:

$$
\varDelta _ { \mathrm { c , i f } } = \lambda N \frac { m + 1 } { m } .\tag{13}
$$

From Proposition 6, assuming a first-order textureless surface:

$$
\kappa _ { \mathrm { I } } ^ { 2 } \propto \frac { 1 } { N ^ { 2 } } \frac { m ^ { 2 } } { \left( m + 1 \right) ^ { 2 } } ,\tag{14}
$$

where throughout this section we use proportionality to absorb constants that do not depend on the lens or sensor. Equation (14) suggests that the in-focus contrast of second-order texture improves with smaller f-number N (larger apertures) and larger reproduction ratio m (larger magnification). However, this analysis does not account for the finite pixel size, which imposes a bound how much we can improve in-focus contrast achieved at an optimal minimum N for any given m. We elaborate next.

We denote by $\varDelta _ { \mathrm { p , 1 } }$ the sensor’s pixel pitch (physical width of pixels on the sensor), and by $\varDelta _ { \mathrm { p } , 0 } : = \left. \Delta _ { \mathrm { p } , 1 } \right/ m$ the magnified pixel pitch (width of the surface area imaged by a pixel after accounting for magnification). Pixels optically blur incident irradiance with a rect kernel of width $\varDelta _ { \mathrm { p } , 0 }$ . This additional blur is an incoherent summation process that will reduce the contrast of subjective speckle when incident irradiance (in object surface-space) varies considerably within the $\varDelta _ { \mathrm { p , 0 } } .$ -sized spatial extent of a pixel—equivalently when the in-focus PSF width $\varDelta _ { \mathrm { c , i f } }$ is significantly smaller than the magnified pixel pitch $\varDelta _ { \mathrm { p , 0 } }$ These considerations suggest that, to maximize the in-focus contrast of secondorder texture at reproduction ratio m, we should set the f-number so that $\varDelta _ { \mathrm { c , i f } } \approx \varDelta _ { \mathrm { p , 0 } }$ . From Equations (13) and (14), the optimal in-focus texture contrast and corresponding optimal f-number become:

$$
\kappa _ { \mathrm { I } } ^ { * 2 } ( m ) \propto \frac { m ^ { 2 } } { \varDelta _ { \mathrm { p , 1 } } ^ { 2 } } \quad \mathrm { a c h i e v e d ~ a t } \quad N ^ { * } ( m ) : = \frac { \varDelta _ { \mathrm { p , 1 } } } { \lambda ( m + 1 ) } .\tag{15}
$$

Further decreasing the f-number results in reduced contrast:

$$
\kappa _ { \mathrm { I } } ^ { 2 } = \kappa _ { \mathrm { I } } ^ { * 2 } ( m ) \frac { \varDelta _ { \mathrm { c , i f } } ^ { 2 } } { \varDelta _ { \mathrm { p , 1 } } ^ { 2 } } \propto \kappa _ { \mathrm { I } } ^ { * 2 } ( m ) \frac { N ^ { 2 } ( m + 1 ) ^ { 2 } } { \varDelta _ { \mathrm { p , 1 } } ^ { 2 } } , \quad \mathrm { i f } \quad N < N ^ { * } ( m ) .\tag{16}
$$

Therefore, Equation (15) shows that using sensors with smaller pixel pitch can improve in-focus contrast of second-order texture, and provides guidance on how to optimally set f-number $N$ at a given target reproduction ratio m.

Lastly, from Proposition 2, characterizing impact on DFF performance requires accounting for both texture contrast and SNR. We follow $\ S \ 5$ and do so by considering how to maximize the texture contrast–SNR product $\Theta ^ { 2 } = \left. \kappa _ { \mathrm { I } } ^ { 2 } \right/ \kappa _ { n } ^ { 2 }$ for a given target reproduction ratio m. We distinguish two scenarios:

1. Exposure time is not constrained: We achieve optimal $\Theta ^ { 2 }$ by setting f-number $N = N ^ { * } ( m )$ (Equation (15)) to maximize texture contrast $\kappa _ { \mathrm { I } } ^ { 2 } .$ , and exposure time t to saturation (Proposition 7) to simultaneously maximize SNR $1 / \kappa _ { n } ^ { 2 }$

2. Exposure time is fixed: Under Poisson noise-limited conditions, $1 / \kappa _ { n } ^ { 2 } \propto 1 / N ^ { 2 }$ (Equation (2)). Thus $\Theta ^ { 2 } \propto \left. m ^ { 2 } \right/ N ^ { 4 } ( m + 1 ) ^ { 2 }$ for $N \geq N ^ { * } ( m )$ (Equation (14)), and $\begin{array} { r } { \Theta ^ { 2 } \propto \dot { \kappa _ { \mathrm { I } } ^ { * 2 } } \dot { ( m ) } ^ { ( m + 1 ) ^ { 2 } } / \Delta _ { \mathrm { p } , } ^ { 2 } } \end{array}$ for $N < N ^ { * } ( m )$ (Equation (16)). Therefore, we achieve optimal $\Theta ^ { 2 }$ by setting f-number N to the minimum value before saturation, collecting as much light as possible through the lens aperture.

Experimental validation. Figure 14 shows experimental results for an indoor scene, using filter bandwidth $\varDelta _ { \lambda } = 1 0$ nm and reproduction ratio $m ~ \approx ~ 1 / 3 0$ . We captured focal stacks using diferent $f -$ number settings, adjusting exposure time to achieve proper exposure at each setting, then visualized the focus measure $\widetilde { \mathrm { c } } _ { j } ^ { 2 }$ at a etextureless patch as a function of f-number. The in-focus contrast of second-order texture, both relative to out-of-focus contrast and in absolute terms, becomes worse as we use more suboptimal settings—f number too large, or too small (pixel pitch limit). The results closely match our theoretical predictions in Equations (14)–(16).

![](images/2e7e51158fcfbda14a90a0fd4f6fa0c1cade73c3fc77f3f7be5cad717e50b1da.jpg)

![](images/57c1ce82602f6234be92917de91edc3edb8742fd52e0c24c00b8130032356f4a.jpg)

![](images/83ca35478aa86d63ad39bec12e1077fcc6b315d04073f27c9dc4798fdfed8f7d.jpg)  
Figure 14: Experimental evaluation of second-order texture contrast in a textureless region (marked, top left) as a function of $f \cdot$ -number. The crops (bottom left) are representative of high and low-contrast cases.

![](images/2d471b2b0058f38bba676e4f5422feea31c7cdab8d71d13175b558f11d05e765.jpg)

![](images/0d840c62e995b2419c391e81d78677f6be6b3f4ebc729ee2d2560729cc10e693.jpg)  
Figure 15: Impact of illumination distance on DFF performance. As the LED moves closer to the scene, the angular bandwidth increases and the coherence area decreases, leading to a reduction in in-focus contrast and DFF performance.

## A.2 Illumination conditions

In § 4.2, we pointed out that the in-focus contrast of second-order texture $\kappa _ { \mathrm { I } } ^ { 2 }$ is dependent on the coherence area $\varDelta _ { \mathrm { i } } ^ { 2 }$ of the illumination (via Q). Our theory therefore predicts that the in-focus contrast of second-order texture, and hence DFF performance, will be worse under wider angular footprints of illumination, which result in smaller coherence areas. This is the only section where we use artificial light sources instead of pre-existing ambient illumination, to facilitate a controlled experiment assessing the impact of coherence area.

Validation of theory. Figure 15 shows an experiment to validate this prediction. To emulate increasing angular bandwidth while keeping all other conditions constant, we moved an LED light source progressively closer to a scene (Figure 16), and captured a full focal stack at each position. As a translation stage moves an LED closer to the scene, the efective angular footprint of incident illumination increases, and the coherence area decreases.

![](images/87d59600c5fe2f7ae5aa129ba71ccf887fd7c08a676f84746926856a5a941ad6.jpg)

The crops show that the in-focus contrast of second-order texture (amplified by introducing a spectral filter) is much weaker when the LED is closer to the scene (20 cm), compared to when the LED is far (60 cm). As our theory predicts, in-focus contrast decreases, and DFF performance worsens.

a translationThis experiment is complementary to the efect of decreasing rail.spectral bandwidth already presented in Figure 8. These two pieces

of evidence together demonstrate that the improvements we see are indeed due to difractive (wave-optical) efects, and therefore can be attributed to the phenomenon of second-order texture that we describe in our theory.

Limitation. This experiment also highlights a limitation of our method in practice: when the angular footprint of illumination is too wide, the texture contrast $\kappa _ { \mathrm { I } } ^ { 2 }$ is still too small after spectral filtering that it cannot overcome the noise level of the camera. Figure 17 shows experiments on a scene comprising objects of diferent levels of geometric complexity (in ascending order: walls, letters, meshed pen holder, hair wig). We illuminate this scene using two artificial sources of illumination, an LED spot light and an LED ring light: The spot light has a small directional bandwidth, and thus large coherence area $\varDelta _ { \mathrm { i } } ^ { 2 } ;$ the ring light has a much larger directional bandwidth, and thus much smaller coherence area $\varDelta _ { \mathrm { i } } ^ { 2 }$ (Equation (9)). The crops show that the in-focus contrast of second-order texture is much weaker under the ring light than under the spot light. As a result, our method provides only minor performance improvements under the ring light, compared to the much more marked improvements under the spot light.

![](images/b542204ee22fe6f8e08a49183996889eb7c7a27fb770a115340d377058b7582a.jpg)  
Figure 17: Using a narrowband spectral filter provides negligible improvements on this scene when we illuminate it with a nearby large ring light. The reason is the excessively small coherence area of the illumination, which results in second-order texture of very low contrast—even when spectrally filtered—relative to the noise level. By contrast, using the filter results in drastic improvements under spot-light illumination, which has a much larger coherence area.

Impact of shadows. In many of our results, shadowed regions do not show as much improvement as non-shadowed regions (e.g. the shadow of the letter in Figure 17). One obvious reason is poor SNR in these regions. Another reason is that shadowed regions are primarily illuminated by indirect light, which typically has a much larger angular footprint than direct light, and thus much smaller coherence area. As a result, second order texture contrast is much weaker in these region, ofering little to no improvement in DFF performance.

## A.3 Material constraints

While we considered the physics of surface scattering, our theory does not account for possible subsurface scattering in the materials of the scene. Subsurface scattering is a phenomenon where light penetrates the surface of a material, scatters internally, and exits at a diferent location. This can significantly reduce the contrast of second-order texture, as it efectively results in the incoherent superposition of far more speckle patterns<sup>6</sup> than would be present in a purely surface-scattering material.

Figure 18 shows experiments on a scene comprising objects made from diferent materials, including materials with varying levels of subsurface scattering (plastic bottle and wax candle on the back right, ceramic cups on the front right). The experiment shows that the performance improvements from our method gradually decrease with increasing amount of scattering (e.g., ceramic cups, crop A), until eventually they become negligible in very translucent materials $( e . g .$ , wax candle, crop C). By contrast, our method provides strong performance improvements at parts of the scene that are largely opaque $( e . g .$ ., the label on the plastic bottle, crop B). This behavior is expected, as subsurface scattering strongly reduces the contrast of subjective speckle—an efect extensively documented and analyzed in optics [46, §6.4.3].

![](images/a64dce638b4d6cfd91e7593b98cb465b428c11545593fa9cad5280722e085968.jpg)  
Figure 18: Performance improvements from using a narrowband filter will vary for diferent optical materials. Crop B, which corresponds to a nearly opaque material (label) shows the strongest improvement. Crop A, which corresponds to a weakly translucent material (ceramic cup) shows reduced but still significant improvement. Crop C, which corresponds to a strongly translucent material (wax) shows negligible improvement. The underlying cause is that strong subsurface scattering in translucent materials (e.g., wax) greatly reduces the contrast of second-order texture (subjective speckle).

## B Interplay between texture contrast and noise

§ 5 provided a high-level overview of the tradeof between second-order texture contrast and SNR for depth recovery performance in terms of the probability of error $\mathrm { P r } \left( \widetilde { \mathrm { c } } _ { \mathrm { i f } } ^ { 2 } < \widetilde { \mathrm { c } } _ { \mathrm { o o f } } ^ { 2 } \right)$ . This section provides an ade editional layer of detail about what happens under the hood–showing how the texturecontrast-SNR product, and the statistics of the focus measure (mean and standard deviation, derived in § H) are impacted by spectral bandwidth and exposure time, and how that, in turn, results in the behaviour of the probability of error shown in Figure 7.

![](images/6ad312f8770398733bf6a1f53e846d680509db3572023759fb916dca4da25ee7.jpg)  
Figure 19: Plot of $\Theta ^ { 2 } ( t , \varDelta _ { k } )$ . Parameters match those in in Figures 7 and 20.

Texture Contrast-SNR product. Just like we visualized the probability of error as a grid as in Figure 7, we can also visualize the texture contrast-SNR product $\Theta ^ { 2 } = \kappa _ { \mathrm { I } } ^ { 2 } / \kappa _ { n } ^ { 2 }$ defined in § 5 directly. An increase in $\Theta ^ { 2 }$ directly maps to a decrease in the probability of error through Lemma 12. Figure 19 shows $\varTheta ^ { 2 }$ as a function of exposure time t for diferent values of spectral bandwidth. Each line stops at the maximum exposure time t that can be used for a given spectral bandwidth, before the camera saturates.

Focus measure statistics. Lemma 11 (from § H) provides approximate expressions for the mean and variance of the sample contrast focus measure ${ \widetilde { \mathrm { c } } } ^ { 2 }$ eWe have the mean and standard deviation of the in-focus and out-of-focus focus measures:

$$
\mathbb { E } \big [ \widetilde { \mathrm { c } } _ { \mathrm { i f } } ^ { 2 } \big ] \approx \kappa _ { \mathrm { I } } ^ { 2 } + \kappa _ { n } ^ { 2 } ,\tag{17}
$$

$$
\mathbb { E } \left[ \widetilde { \mathrm { c } } _ { \mathrm { o o f } } ^ { 2 } \right] \approx \kappa _ { n } ^ { 2 } ,\tag{18}
$$

$$
\sqrt { \mathbb { V } [ \widetilde { \mathrm { c } } _ { \mathrm { i f } } ^ { 2 } ] } \approx \sqrt { \frac { 2 } { T - 1 } } \big ( \kappa _ { \mathrm { I } } ^ { 2 } + \kappa _ { n } ^ { 2 } \big ) ,
$$

$$
\sqrt { \mathbb { V } [ \widetilde { \mathrm { c } } _ { \mathrm { o o f } } ^ { 2 } ] } \approx \sqrt { \frac { 2 } { T - 1 } } ( \kappa _ { n } ^ { 2 } ) .\tag{19}
$$

As a consequence, the mean separation between the (noisy) in-focus and out-offocus sample contrast focus measures

$$
\mathbb { E } \big [ \widetilde { \mathrm { c } } _ { \mathrm { i f } } ^ { 2 } \big ] - \mathbb { E } \big [ \widetilde { \mathrm { c } } _ { \mathrm { o o f } } ^ { 2 } \big ] \approx \kappa _ { \mathrm { I } } ^ { 2 }\tag{20}
$$

In addition, we know from Proposition 6 that for a first-order textureless surface $( \kappa _ { \overline { { \mathrm { E } } } } ^ { 2 } = 0 )$

$$
\kappa _ { \mathrm { I } } ^ { 2 } = { \frac { 1 } { M Q } }\tag{21}
$$

and from Equation (2)

$$
\kappa _ { n } ^ { 2 } = \frac { 1 } { t \zeta \mu _ { \Phi } } \biggl ( 1 + \frac { \sigma _ { \mathrm { r e a d } } ^ { 2 } } { t \zeta \mu _ { \Phi } } \biggr )\tag{22}
$$

Thus, given the parameters of the capture system, we can plug in these values to compute the mean and standard deviation of the in-focus and out-of-focus focus measures. The bottom row in Figure 20 plots these quantities as an extension of Figure 7. The ‘theory’ curves show the predicted separation $\mathbb { E } \left[ \widetilde { \mathrm { c } } _ { j } ^ { 2 } \right] - \kappa _ { n } ^ { 2 }$ for $j = j _ { \mathrm { i f } }$ and $j = j _ { \mathrm { o o f } }$ e(the latter is uniformly zero, as evident from Equation (18)), also labeling a region 1 standard deviations surrounding it. These regions give us a sense of the width of the distributions of $\widetilde { \mathrm { c } } _ { \mathrm { i f } } ^ { 2 }$ and $\widetilde { \mathrm { c } } _ { \mathrm { o o f } } ^ { 2 }$ . The separation between the e ein-focus and out-of-focus focus measure distributions provides immediate insight into DFF performance. When the distributions of $\widetilde \mathrm { c } _ { \mathrm { i f } } ^ { 2 }$ and $\widetilde { \mathrm { c } } _ { \mathrm { o o f } } ^ { 2 }$ are well-separated, e ethe chances of a patch being misclassified as in-focus or out-of-focus is low, and hence $\mathrm { P r } \big ( \widetilde { \mathrm { c } } _ { \mathrm { i f } } ^ { 2 } < \widetilde { \mathrm { c } } _ { \mathrm { o o f } } ^ { \bar { 2 } } \big )$ is low. When they are close together, the opposite is true e e(worst case: 50-50 chance).

![](images/0fce2d38929c0cb0cb647003a08f23d4bea46d3a04ab229ca5ff2665e56e52b5.jpg)  
Figure 20: Visualization of the mean and standard deviation of the focus measure (specifically, $\mathbb { E } \left\lceil \widetilde { \mathrm { c } } ^ { 2 } \right\rceil - \kappa _ { n } ^ { 2 }$ , and $\sqrt { \mathbb { V } [ \widetilde { \mathrm { c } } ^ { 2 } ] } )$ ) along the three paths shown in Figure 7. ‘theory’ corresponds to direct analytical evaluation using Equations (17)–(19), while ‘MC corresponds to Monte-Carlo estimates of the same quantities using samples from simulation. The MC estimates (using 10k samples per point) have higher variance in regimes of very low exposure.

This plots provide a clearer picture of how capture parameters (exposure time and spectral bandwidth) individually influence the statistics of the focus measures in and out of focus with more granularity.

1. If the spectral bandwidth is reduced at a fixed exposure time, the mean separation between the in-focus and out-of-focus focus measures increases in accordance with Equation (20) (right sections).

2. When the exposure time is increased (left section), the mean separation is unafected, but the standard deviation of the focus measures decreases, reducing overlap between the distributions and thus reducing the probability of error.

3. If the spectral bandwidth is reduced, while also increasing the exposure time to maintain proper exposure (middle section), the mean separation increases, while the in-focus standard deviation increases slightly (in accordance with Equation (19)).

Match with Monte-Carlo simulation. Overlaid on the theory-based curves and shaded regions (from Equations (19)–(20)) are the corresponding Monte-Carlo (MC) estimates from simulation. The close match between theory and simulation (for both focus measure statistics, and probability of error) shows that our theoretical analysis is accurate, and that the assumptions made in § H are reasonable. In particular, we assumed that the distributions of $\widetilde { \mathrm { c } } _ { \mathrm { i f } } ^ { 2 }$ and $\widetilde { \mathrm { c } } _ { \mathrm { o o f } } ^ { 2 }$ eare approximately Gaussian, which is validated by the close match between theory and simulation. One may notice that the theoretical curves consistently overestimate the mean contrast in the high-contrast regimes (left and middle plots). This can be directly attributed to an assumption we make in the derivation of Lemma 11, i.e. we assume that the intensities of pixels are independent. See § H for details.

Table 1: Fraction of unrecovered pixels η with $\xi > \xi _ { \mathrm { t h } } ( = 4 . 0 )$ for all scenes in our results (main paper and supplement). Dashes indicate unavailable (not captured) datasets. Visit https://imaging.cs.cmu.edu/second-order-texturefor interactive visualizations of all scenes
<table><tr><td></td><td>scene no filter</td><td>100 nm</td><td>48 nm</td><td>25 nm</td><td>10 nm</td></tr><tr><td>illusion</td><td>0.834</td><td></td><td></td><td></td><td>0.004</td></tr><tr><td>side table</td><td>0.560</td><td>0.469</td><td>一</td><td></td><td>0.243</td></tr><tr><td>helmet</td><td>0.742</td><td></td><td></td><td>0.461</td><td>0.313</td></tr><tr><td>knight</td><td>0.866</td><td>0.532</td><td>0.427</td><td>0.388</td><td>0.362</td></tr><tr><td>set cards</td><td>0.842</td><td>0.732</td><td>0.621</td><td>0.460</td><td>0.287</td></tr><tr><td>bin</td><td>0.853</td><td></td><td></td><td>0.461</td><td>0.583</td></tr><tr><td>sculpture</td><td>0.552</td><td></td><td></td><td></td><td>0.113</td></tr><tr><td>corner</td><td>0.715</td><td>0.645</td><td>0.615</td><td>0.613</td><td>0.644</td></tr><tr><td>columns</td><td>0.823</td><td>0.664</td><td>0.668</td><td>0.631</td><td>0.642</td></tr><tr><td>trash can</td><td>0.633</td><td>0.493</td><td>0.501</td><td>0.462</td><td>0.216</td></tr><tr><td>hair &amp; mesh</td><td>0.561</td><td>0.423</td><td>0.359</td><td>0.261</td><td>0.141</td></tr><tr><td>bottles &amp; mugs</td><td>0.534</td><td>0.415</td><td>0.371</td><td>0.348</td><td>0.295</td></tr><tr><td>cup</td><td>0.893</td><td>1</td><td>一</td><td>-</td><td>0.518</td></tr></table>

## C Depth recovery performance evaluation

We report quantitative metrics across all our scenes, and validate our z-scorebased evaluation metric. Visit our webpage (https://imaging.cs.cmu.edu/secondorder-texture) for interactive visualizations of all scenes in the main paper and supplement, including depth maps, surface renders, and comparisons.

## C.1 Quantitative comparison

Table 1 aggregates DFF performance metrics across all captured scenes we show in both the main paper and the supplement. As we do not have ground-truth depth, to assess performance, we use the fraction of unrecovered pixels η we defined in $\ S \ 6 ,$ which serves as a measurable proxy for the probability of error $\mathrm { P r } ( \widetilde { \mathrm { c } } _ { \mathrm { i f } } ^ { 2 } < \widetilde { \mathrm { c } } _ { \mathrm { o o f } } ^ { 2 } )$ (thus lower values of η represent better DFF performance). The e etable shows that, consistently across all scenes, performance improves as spectral bandwidth decreases—and while adjusting exposure time to maintain proper exposure—in agreement with our theoretical predictions. The main paper shows quantitative comparisons also for cases of underexposure (Figure 10).

![](images/3ab0b9ec3bcd6aad0234103b51e63b644911d9ec33d474a1e34292f04e26e68b.jpg)

![](images/0e2b8d0c70ebee1a672e4bbec34ec98c677933be742ef82b4a70a41bb4c171d6.jpg)

![](images/51e56f22f438e9d3b5a2ab381806d53c9e9d94d527649185530b7b2c5811495b.jpg)

![](images/5c32ae5db2dc9fec84e656be56b42b3b51cba532113ef5cf230eae7e7f9c4913.jpg)  
Figure 21: Validation of the η metric using RMSE. (scenes captured indoors). For the reference depth map, a projector was used to illuminate the scene with a fine checkerboard pattern to create artificial texture.

Table 2: Spectral filters we use for experiments.
<table><tr><td>filter model</td><td>center wavelength (nm)</td><td>full-width half-max (FWHM) (nm)</td></tr><tr><td>EO #65-216</td><td>532</td><td>10</td></tr><tr><td>EO #84-115</td><td>534</td><td>25</td></tr><tr><td>EO #67-045</td><td>534.5</td><td>48</td></tr><tr><td>EO #33-331</td><td>550</td><td>100</td></tr></table>

## C.2 Validation of z-score-based metric with active DFF reference

Figure 21 shows comparisons with a reference from active DFF—a projector illuminates the scene with a high-frequency pattern to create artificial texture. We compute RMSE relative to this reference depth, excluding pixels with lowconfidence reference.

We observe: 1. RMSE decreases as spectral bandwidth narrows, validating our theory. 2. RMSE is strongly correlated with our η metric (fraction of unrecovered pixels). This strong correlation is expected: From DFF theory [9, 68], wrong depth estimates with high z-score are extremely unlikely (a patch that, when out of focus, serendipitously has far higher noise than at other focus settings). Essentially all wrong depth estimates arise from flat focus-measure curves (low texture or high noise), and hence have low z-score. Therefore, our η metric quantifies DFF performance as accurately as RMSE, while being much more practical for outdoor experiments. Capturing active DFF for reference is challenging under sunlight.

## D Implementation details

We provide details for simulations and experiments in the main paper and supplement.

## D.1 Capture setup

Our setup comprises only of-the-shelf components for standard depth from focus. We use a machine vision camera (FLIR Black-Fly S BFS-U3-122S6M-C) with a monochromatic sensor (Sony IMX304, 1.4 cm diagonal, 3.5 µm pixel pitch, 12 bits). The camera provides access to RAW measurements. We equip the camera with an electronic lens mount (ISSI Canon EF lens controller) for programmatic focus and aperture control,

![](images/b996f6ef73d9d3c7c80013ef5149dd2b390c2658ee278894751dab8703a2ee2e.jpg)  
Figure 22: Capture setup.

and a photographic lens (Canon EF-S 60mm f/2.8 Macro USM) for imaging. We use standard filter rings and an optical filter wheel to mount and easily rotate filters at the front of the lens. Lastly, we use narrowband spectral filters of 2 inch diameter (Edmund Optics, diferent FWHM bandwidths, center wavelength 530 nm). Figure 22 shows a photograph, and Table 2 lists all filters we use in our experiments.

Algorithm 1 Depth recovery using single-step aggregation.   
1: procedure SingleStepDFF $( \left\{ \widetilde { \mathrm { I } } _ { j } \right\} _ { j = 1 } ^ { J } )$   
2: for $j \in \{ 1 , \dotsc , J \}$ do   
3: $\widetilde { \mathrm { c } } _ { j } ^ { 2 } \gets \mathrm { L o c a L F M } ( \widetilde { \mathrm { I } } _ { j } )$ \triangleright focus measure stack   
4: $\widetilde { \mathrm { c } } _ { j } ^ { 2 } \gets \widetilde { \mathrm { c } } _ { j } ^ { 2 } * g _ { \sigma _ { \mathrm { a g g } } }$ \triangleright Gaussian kernel aggregation   
5: end for   
6: $j ^ { \star } ( \vec { p } ) \gets \mathrm { a r g m a x } _ { j } \widetilde \mathrm { c } _ { j } ^ { 2 } ( \vec { p } )$   
7: $d \gets \mathrm { I N T E R P D E P T H } ( \widetilde { \mathrm { c } } . , j ^ { \star } )$   
8: $\xi \gets \mathrm { Z s c o R E } ( \widetilde { \mathrm { c } } _ { \cdot } ^ { 2 } , j ^ { \star } )$ \triangleright pixel-wise z-scores   
9: return $d , \xi$   
10: end procedure   
11: procedure $\mathrm { L o c A L F M } ( \widetilde { \mathrm { I } } _ { j } )$   
12: return ${ \textstyle \left[ \widetilde { \mathrm { I } } _ { j } * \mathrm { L } _ { 3 \times 3 } \right] ^ { 2 } }$ \triangleright normalized squared Laplacian   
$\scriptstyle \overbrace { { \left[ \tilde { \ I } _ { j } * \mathbf { U } _ { 3 \times 3 } \right] } ^ { 2 } }$   
13: end procedure   
14: procedure $\mathrm { Z s c o R E } ( \widetilde { \mathrm { c } } _ { \cdot } ^ { 2 } , j ^ { \star } )$   
15: $\widetilde { \mathrm { c } } _ { \mathrm { m e d } } ^ { 2 } ( \overrightarrow { p } ) = \mathrm { m e d i a n } _ { j } \widetilde { \mathrm { c } } _ { j } ^ { 2 } ( \overrightarrow { p } )$   
16: MAD(<sup>#–</sup>p ) = median<sub>j</sub> $\left| \widetilde { \mathrm { c } } _ { j } ^ { 2 } ( \xrightarrow [ p ] { } ) - \widetilde { \mathrm { c } } _ { \mathrm { m e d } } ^ { 2 } ( \xrightarrow [ p ] { } ) \right|$   
17: return $\frac { \widetilde { \mathbf { c } } _ { j \star } ^ { 2 } - \widetilde { \mathbf { c } } _ { \mathrm { m e d } } ^ { 2 } } { \mathbf { M A D } }$ \triangleright pixel-wise robust z-score   
18: end procedure

## D.2 Depth recovery algorithms

Algorithms 1 and 2 detail the methods we use in $\ S \ O 6$ for recovering depth from a focal stack. Algorithm 1 performs one aggregation over a patch of fixed size to compute a Laplacian-based focus measure. Algorithm 2 is a pixel-adaptive variant: At each pixel, it performs multiple consecutive aggregations—each progressively increasing the efective patch size—until the robust z-score at that pixel reaches a certain threshold. Thus this algorithm varies per-pixel patch size to adapt to areas of diferent texture contrast. In both algorithms, the routine InterpDepth uses three-point Gaussian interpolation of the focus measure to determine a peak dithered between focus indices [9].

## D.3 Monte Carlo simulation

Our Monte Carlo simulation is a direct implementation of the description of second-order texture in $\ S \ F$ and the full noise model in $\ S \ G$ . Below, we explain the sequence of steps alongside implementation details and references to corresponding equations in the supplement and main text. We use a pseudorandom number generator for all sampling steps. Algorithm 3 summarizes the full Monte Carlo pipeline.

1. We simulate a surface that is fronto-parallel to the camera. We model its texture using the blurred outgoing radiance $\mathcal { L } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } )$ (Equation (73)) as a two-dimensional grid of spacing $\varDelta _ { \mathrm { i } }$ <sub>i</sub>, where $\left\{ \overrightarrow { q } _ { m } \right\}$ is the set of grid centers (based on Proposition 10 in § F).

Algorithm 2 Pixel-adaptive depth recovery using progressive aggregation.   
1: procedure PixelAdaptiveDFF $( \left\{ \widetilde { \mathrm { I } } _ { j } \right\} _ { j = 1 } ^ { J } , \xi _ { \mathrm { t h } } )$   
2: $\xi ( \vec { p } )  1 ; d ( \vec { p } )  \mathrm { N a N } ; \chi ( \vec { p } ) $ false \triangleright init. z-scores, depth, completion   
3: $l  0$   
4: for $j \in \{ 1 , \dotsc , J \}$ do   
5: $\widetilde { \mathrm { c } } _ { j } ^ { 2 } \gets \mathrm { L o c a L F M } ( \widetilde { \mathrm { I } } _ { j } )$ \triangleright init. focus measure stack   
6: end for   
7: while $l < l _ { \mathrm { m a x } }$ and $\neg \mathrm { a l l } ( \chi )$ do   
8: for $j \in \{ 1 , \ldots , J \}$ do   
9: $\widetilde { \mathbf { c } } _ { j } ^ { 2 } \gets \mathrm { W }$ eightedAggStep $( \widetilde { \mathbf { c } } _ { j } ^ { 2 } , \xi )$   
10: end for   
11: $j ^ { \star } ( \vec { p } ) $ argmax $_ j \widetilde { \mathbf { c } } _ { j } ^ { 2 } ( \overrightarrow { p } )$   
12: ξ ← Zscore $\therefore ( \widetilde { \mathsf { c } } _ { \cdot } ^ { 2 } , \dot { j } ^ { \star } )$ \triangleright update pixel-wise z-scores   
13: χ<sub>new</sub> $ \xi > \xi _ { \mathrm { t h } } \wedge \lnot \chi$   
14: $d [ \chi _ { \mathrm { n e w } } ] \gets .$ InterpDepth $( \widetilde { \mathrm { c } } _ { \cdot } ^ { 2 } , j ^ { \star } ) [ \chi _ { \mathrm { n e w } } ]$   
15: $\chi  \chi \lor$ χ<sub>new</sub>   
16: $l  l + 1$   
17: end while   
18: return d   
19: end procedure   
20: procedure WeightedAggStep $( \widetilde { \mathbf { c } } _ { j } ^ { 2 } , \boldsymbol { \xi } )$   
21: return $\frac { \left[ \left( \widetilde { \mathbf { c } } _ { j } ^ { 2 } \cdot \boldsymbol { \xi } \right) * g _ { \sigma _ { 1 } } \right] } { \left[ \boldsymbol { \xi } * g _ { \sigma _ { 1 } } \right] }$ \triangleright z-score weighted aggregation   
22: end procedure

2. We consider first-order textureless appearance, $i . e . , \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) }$ is constant across grid locations, or $\kappa _ { \overline { { \mathcal { L } } } } ^ { 2 } ( \lrcorner ) = \kappa _ { \overline { { \mathrm { E } } } } ^ { 2 } = 0$ . Additionally, we assume that $\overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) }$ at each location is proportional to spectral bandwidth $\varDelta _ { k }$ according to Equation (159), to model brightness attenuation due to narrowing bandwidth; thus: $\overline { { \mathcal { L } } } ^ { ( \Delta _ { \mathrm { i } } ) } =$ $( \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ) _ { \mathrm { o r i g } } \mathrm { g _ { p } } \varDelta _ { k }$ , where $( \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ) _ { \mathrm { o r i g } }$ is the original value of $\overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) }$ without spectral filtering.

3. We sample each cell of the coherence area grid independently from a Gamma distribution according to Proposition 10. We report related parameters in Table 3a.

4. We compute the irradiance reaching the sensor at the in-focus setting according to the discrete convolution of the PSF $\mathrm { P _ { c } }$ with the outgoing radiance grid $\mathcal { L } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } )$ in Equation (77). We model the PSF as a discrete Gaussian kernel with standard deviation $\left. \Delta _ { \mathrm { c } } \right/ _ { 2 }$ as in Equation (113). We compute the in-focus PSF width in object space using Equation (13). We scale the irradiance at the sensor by a factor $\propto { m ^ { 2 } } / { N _ { \mathrm { e f f } } ^ { 2 } }$ to model how the energy reaching the sensor through the lens varies as a function of m and N. We report related parameters in Table 3b.

```latex
Algorithm 3 Monte Carlo focus measure sampler
default parameters $g , \sigma _ { \mathrm { p r e } } , \sigma _ { \mathrm { p o s t } }$ etc. from Tables 3a–3c
Pixel resolution $H \times W .$
1: procedure MCFMSampler $( \varDelta _ { k } , t )$
2: $\overline { { \mathcal { L } } } ^ { ( \Delta _ { \mathrm { i } } ) }  \mathrm { g _ { p } } \Delta _ { k } \cdot \overline { { \mathcal { L } } } ^ { ( \Delta _ { \mathrm { i } } , \mathrm { o r i g } ) }$ \triangleright Equation (171)
3: $\begin{array} { r } { \mathcal { L } ^ { ( \Delta _ { \mathrm { i } } ) } ( \vec { q } _ { m } ) \gets \mathrm { G a m m a } \Big ( M , \frac { M } { \overline { { \mathcal { L } } } ^ { ( \Delta _ { \mathrm { i } } ) } } \Big ) } \end{array}$ \triangleright Equation (82)
4: $\mathrm { E _ { i f } }  \mathrm { P S F B L U R } \big ( \mathcal { L } ^ { ( \Delta _ { \mathrm { i } } ) } \big )$
5: $\widetilde { \mathrm { I } } _ { \mathrm { i f } } ( \overrightarrow { p } )  \mathrm { S E N S O R R E A D O U T } ( \mathrm { E } _ { \mathrm { i f } } , t ) .$
6: $\widetilde { \mathrm { c } } _ { \mathrm { i f } } ^ { 2 } ( \overrightarrow { p } ) \gets \mathrm { S A M P L E C o N T R A S T F M } ( \widetilde { \mathrm { I } } _ { \mathrm { i f } } )$ \triangleright in-fo cus sample
7: ec<sup>2</sup> f(<sup>#–</sup>p ) <sup>←</sup> <sup>S</sup>ample<sup>C</sup>ontrast<sup>FM</sup>(µI <sup>·</sup> ones $( H , W ) )$ \triangleright out of focus sample
8: return $\widetilde { \mathbf { c } } _ { \mathrm { i f } } ^ { 2 } , \widetilde { \mathbf { c } } _ { \mathrm { o o f } } ^ { 2 }$
9: end procedure
10: procedure $\mathrm { P S F B L U R } ( \mathcal { L } ^ { ( \varDelta _ { \mathrm { i } } ) } )$
11: $\begin{array} { r } { \mathrm { E } ( \vec { q } )  \sum _ { m } { \varDelta _ { \mathrm { i } } ^ { 2 } \mathrm { P } _ { \mathrm { c } } ( \vec { q } - \vec { q } _ { m } ) \mathcal { L } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \vec { q } _ { m } ) } } \end{array}$ \triangleright Equation (77)
12: return E
13: end procedure
14: procedure SensorReadout(E, t)
15: $\Phi ( \vec { p } )  \mathrm { E } ( \vec { p } ) \cdot \varDelta _ { \mathrm { p } } ^ { 2 }$ \triangleright sample irradiance coarsely at pixel resolution
16: P ← Poisson $( ( \zeta \Phi + D ) t )$
17: $Z _ { \mathrm { p r e } } \gets \mathcal { N } \big ( 0 , \tilde { \sigma } _ { \mathrm { p r e } } ^ { 2 } \big ) ; Z _ { \mathrm { p o s t } } \gets \mathcal { N } \big ( 0 , \sigma _ { \mathrm { p o s t } } ^ { 2 } \big )$
18: $\widetilde \mathrm { I } ( \overrightarrow { p } ) \gets \operatorname* { m i n } ( \lfloor \lambda / g ( \operatorname* { m i n } ( P , P _ { \operatorname* { m a x } } ) + Z _ { \mathrm { p r e } } ) + Z _ { \mathrm { p o s t } } \rfloor , \mathrm { I } _ { \operatorname* { m a x } } )$ \triangleright Equation (120)
19: return eI
20: end procedure
21: procedure SampleContrastFM(eI)
22: return $\frac { \widetilde { \mathbf { s } } ^ { 2 } } { \widetilde { \mathbf { m } } ^ { 2 } }$ \triangleright Equation (1)
23: end procedure
```

5. We sum the resulting irradiance grid $\operatorname { E } ( { \overrightarrow { p } } )$ —now mapped to image space—over the pixel area $\varDelta _ { \mathrm { p } } \times \varDelta _ { \mathrm { p } }$ to get the flux $\Phi ( \vec { p } )$ at each pixel at the camera’s resolution.

6. Given the flux at each pixel, we use the full noise model of Equation (120) to sample the noisy intensity $\widetilde { \mathrm { I } } ( \overrightarrow { p } )$ at each pixel. We report related parameters in Table 3c.

7. Given the noisy intensity measurements, we compute the focus measure ${ \widetilde { \mathrm { c } } } ^ { 2 }$ (§ 3) for a square-shaped patch of size ${ \sqrt { T } } \times { \sqrt { T } } \ ( 5 \times 5 )$ e centered at the pixel of interest.

8. We repeat this process multiple times to obtain Monte-Carlo estimates of P $\mathrm { r } \left( \widetilde { \mathrm { c } } _ { \mathrm { i f } } ^ { 2 } > \widetilde { \mathrm { c } } _ { \mathrm { o o f } } ^ { 2 } \right)$ , for diferent values of $\varDelta _ { k }$ and t. We plot these estimates in eFigure 7.

Table 3: Parameters for Monte Carlo simulation in Figure 7.  
(a) Second-order texture parameters.  
(b) Camera lens settings.
<table><tr><td>parameter value</td><td></td><td>description</td></tr><tr><td> $\sigma _ { \mathrm { r } }$ </td><td>3µm</td><td>RMS height of surface</td></tr><tr><td> $\varDelta _ { \mathrm { i } }$ </td><td></td><td>12 µm width of coherence cell grid</td></tr><tr><td> $\bar { \lambda }$ </td><td>532 nm</td><td>central wavelength</td></tr></table>

<table><tr><td>parameter value</td><td>description</td></tr><tr><td>m</td><td>0.01 reproduction ratio</td></tr><tr><td>N 7</td><td>f-number</td></tr></table>

(c) Camera sensor parameters.
<table><tr><td>parameter</td><td>value</td><td>description</td></tr><tr><td> $g$ </td><td>10e−/DN</td><td>sensor gain</td></tr><tr><td> $\sigma _ { \mathrm { p r e } }$ </td><td> $1 0 \mathrm { e } ^ { - }$ </td><td>pre-amp noise variance</td></tr><tr><td> $\sigma _ { \mathrm { p o s t } }$ </td><td>15 DN</td><td>post-amp noise variance</td></tr><tr><td> $\mathrm { l o g _ { 2 } ( I _ { m a x } ) }$ </td><td>12</td><td>ADC bit depth</td></tr><tr><td> $\boldsymbol { v }$ </td><td>0.4</td><td>quantum efficiency</td></tr><tr><td> $D$ </td><td> $1 . 0 \mathrm { e } ^ { - } / \mathrm { s }$ </td><td>dark current</td></tr><tr><td> $P _ { \mathrm { m a x } }$ </td><td> $3 5 0 0 0 \mathrm { e } ^ { - }$ </td><td>full-well capacity</td></tr><tr><td> $\varDelta _ { \mathrm { p } }$ </td><td>3.45 µm</td><td>pixel pitch</td></tr></table>

## E Appearance model for second-order texture

We present a derivation of Proposition 4 consistent with physical-optics principles for an explicit micro-geometry of the surface, mapping wave quantities to conventional radiometric quantities used in computer vision. We then explain how traditional BRDF-based image formation models used in computer vision are a first-order abstraction of appearance based on the statistics of the surface heightfield, and then quantify the strength of second-order texture in $\ S \ F$ . Our formulations largely follow Goodman [46, Sections 5.8 & 6.3] and Steinberg and Yan [62].

## E.1 Incident illumination model

Consider the 2D plane macroscopically tangent to the surface of the object at the point of interest with macroscopic outward normal nˆ (Figure 3). At any point $\bar { \boldsymbol { \rho } } \in \mathbb { R } ^ { 2 }$ in this plane, we model the incoming illumination as consisting of a superposition of monochromatic plane waves of varying direction and wavelength:

$$
\mathrm { U _ { i } } ( \vec { \rho } ) = \int _ { k \in \mathbb { R } ^ { + } } \int _ { \mathcal { H } ^ { 2 } ( \hat { n } ) } \mathcal { G } ( \hat { \omega } , k ) e ^ { \mathrm { j } k \hat { \omega } _ { x y } \cdot \vec { \rho } } \mathrm { d } \sigma ( \hat { \omega } ) \mathrm { d } k\tag{23}
$$

Here, $\hat { \omega }$ is the outward unit vector antiparallel to the direction of the incoming plane wave, $k = 2 \pi / \lambda$ represents its (angular) wavenumber, and

$$
\mathcal { H } ^ { 2 } ( \hat { n } ) = \left\{ \hat { \omega } \in \mathbb { R } ^ { 3 } , \| \hat { \omega } \| = 1 , \hat { \omega } \cdot \hat { n } > 0 \right\}\tag{24}
$$

is the upper-half space of valid outward incident direction vectors. $\hat { \omega } _ { x y }$ is the component of ωˆ parallel to the plane, $\boldsymbol { i . e . , \hat { \omega } = [ \hat { \omega } _ { x y } , \ \omega _ { z } \hat { n } ] }$

In the context of computer vision with direct and indirect illumination, one may consider these plane waves to have been generated by a variety of light sources in the scene. The amplitude of these plane waves $( \hat { \omega } , k )$ are modeled to be a result of the random process $\{ \mathcal { G } : \mathcal { H } ^ { 2 } ( \hat { n } ) \times \mathbb { R } ^ { + }  \mathbb { C } \}$ (Units: $\lceil \mathrm { W } ^ { 0 . 5 } / \mathrm { s r } \rceil )$ . Any observable radiometric quantity such as irradiance or intensity is to be computed by averaging the result over this ensemble. [71]

Mutual incoherence assumption. We assume these plane waves to be mutually incoherent, i.e. $\mathcal { G } ( \hat { \omega } , \boldsymbol { k } )$ are uncorrelated for any non-identical pair $( \hat { \omega } , k )$ , that is,

$$
\langle \mathcal { G } ( \hat { \omega } _ { 1 } , k _ { 1 } ) \mathcal { G } ^ { * } ( \hat { \omega } _ { 2 } , k _ { 2 } ) \rangle = \mathrm { R } _ { \mathrm { i } } ( \hat { \omega } _ { 1 } , k _ { 1 } ) \delta _ { k } ( k _ { 1 } - k _ { 2 } ) \delta _ { \hat { \omega } } ( \hat { \omega } _ { 1 } - \hat { \omega } _ { 2 } )\tag{25}
$$

Here, $\langle \cdot \rangle$ represents an average<sup>7</sup> over the ensemble ${ \mathcal { G } } _ { : }$ , and $\mathrm { R } _ { \mathrm { i } } ( \hat { \omega } _ { 1 } , k _ { 1 } )$ equals the foreshortened radiance

$$
\operatorname { R i } ( { \hat { \omega } } , k ) = \operatorname { L i } ( { \hat { \omega } } , k ) ( { \hat { \omega } } \cdot { \hat { n } } ) .\tag{26}
$$

$\mathrm { L } _ { \mathrm { i } } ( \hat { \omega } , \boldsymbol { k } ) \ \mathrm { ( U n i t s : / W m ^ { - 1 } s r ^ { - 1 } } ] \big )$ is the incoming spectral radiance environment map introduced in $\ S \ 4 .$ . The assumption of mutual incoherence allows us make a correspondence between plane wave amplitudes with incoming radiance as defined in radiometry. In addition, we assume the environment map is separable in k and ωˆ, i.e. $\operatorname { L } _ { \mathrm { i } } ( \hat { \omega } , \boldsymbol { k } ) = \ell _ { \mathrm { i } } ( \boldsymbol { k } ) \cdot \operatorname { L } _ { \mathrm { i } } ( \boldsymbol { \hat { \omega } } )$ , where the normalized spectral profile is $\ell _ { \mathrm { i } } ( k )$ , which integrates to unity.

The above is a suitable model for real-world ambient illumination, where plane waves from diferent incident directions due to a variety of far-field light sources $( e . g$ . sun, lamps, ceiling lamps, indirect reflections) and wavenumbers are statistically uncorrelated due to randomness of the emission process at each point on the light source(s). By considering the plane-wave amplitudes $\mathcal { G }$ to be position-independent, we have thus described an environment lighting model. This is suficient for our purposes since we only seek to describe the scattering interaction with the surface in the local neighborhood at the point of interest.

We re-express the complex incident field created by the superposition of these plane waves (Equation (23)) as

$$
\mathrm { U _ { i } } ( \vec { \rho } ) = \int _ { k } \mathrm { U _ { i } } ( \vec { \rho } , k ) \mathrm { d } k\tag{27}
$$

$$
\mathrm { U _ { i } } ( \vec { \rho } , k ) = \int _ { \mathcal { H } ^ { 2 } ( \hat { n } ) } \mathcal { G } ( \hat { \omega } , k ) e ^ { \mathrm { j } k \hat { \omega } _ { x y } \cdot \vec { \rho } } \mathrm { d } \sigma ( \hat { \omega } )\tag{28}
$$

Example: incident irradiance. We now provide a simple example of computing an observable radiometric quantity using the above illumination model. The incident irradiance $\mathrm { ( U n i t s { : } [ W m ^ { - 2 } ] ) }$ at a point on the local 2D plane is given by the average magnitude-squared of the incident field:[72]

$$
\mathrm { E } _ { \mathrm { i } } ( \vec { \rho } ) = \left. \left| \mathrm { U } _ { \mathrm { i } } ( \vec { \rho } ) \right| ^ { 2 } \right.\tag{29}
$$

$$
= \int _ { \mathbb { R } ^ { + } } \left. \left| \mathrm { U } _ { \mathrm { i } } ( \vec { \rho } , k ) \right| ^ { 2 } \right. \mathrm { d } k\tag{30}
$$

$$
= \int _ { \mathbb { R } ^ { + } } \operatorname { E i } ( { \vec { \rho } } , k ) \mathrm { d } k .\tag{31}
$$

The cross-terms vanish since $\mathcal { G } ( \cdot , \cdot )$ is uncorrelated for non-identical wavenumbers. We now similarly expand the spectral density of incident irradiance $\operatorname { E } _ { \mathrm { i } } ( { \vec { \rho } } , k )$ for wavenumber $k ,$ as $\mathrm { ( U n i t s : \lceil W m ^ { - 1 } \rceil ) }$ )

$$
\mathrm { E } _ { \mathrm { i } } ( \vec { \rho } , \boldsymbol { k } ) = \left. \left| \mathrm { U } _ { \mathrm { i } } ( \vec { \rho } , \boldsymbol { k } ) \right| ^ { 2 } \right.\tag{32}
$$

$$
= \int _ { \mathcal H ^ { 2 } ( \hat { n } ) } \int _ { \mathcal H ^ { 2 } ( \hat { n } ) } \langle \mathcal G ( \hat { \omega } _ { 1 } , k ) \mathcal G ^ { * } ( \hat { \omega } _ { 2 } , k ) \rangle e ^ { \mathrm { j } k \overrightarrow { \rho } \cdot ( \hat { \omega } _ { x y , 1 } - \hat { \omega } _ { x y , 2 } ) } \mathrm { d } \sigma ( \hat { \omega } _ { 1 } ) \mathrm { d } \sigma ( \hat { \omega } _ { 2 } )\tag{33}
$$

$$
= \ell _ { \mathrm { i } } ( k ) \int _ { \mathcal { H } ^ { 2 } ( \hat { n } ) } \mathrm { L } _ { \mathrm { i } } ( \hat { \omega } ) ( \hat { \omega } \cdot \hat { n } ) \mathrm { d } \sigma ( \hat { \omega } )\tag{34}
$$

which is independent of position ${ \vec { \rho } } ;$ , as expected, since we use an environment map. Hence, we can drop the position dependence and use $\mathrm { E _ { i } } ( k )$ and $\mathrm { E _ { i } }$ instead. In summary, thanks to the assumption of mutual incoherence of the incident plane waves, the above example illustrates that for the computation of observable radiometric quantities, we can consider the efect of each incident plane wave $( \hat { \omega } , k )$ independently of others (using $\mathrm { L } _ { \mathrm { i } } ( \hat { \omega } , \boldsymbol { k } ) )$ and then sum up their contributions linearly.

## E.2 Coherence function of the incident illumination

Before proceeding to compute the scattered field and irradiance on the sensor of the camera, we introduce the following definition (Equation (9) in main text):

## Definition 8: Coherence Function

We define the coherence function $\varLambda _ { \mathrm { i } } ( \cdot )$ of the illumination environment map $\mathrm { L } _ { \mathrm { i } } ( \cdot )$ as its projected Fourier transform:

$$
\varLambda _ { \mathrm { i } } ( \vec { \rho } ; k ) : = \int _ { \mathcal { H } ^ { 2 } ( \hat { n } ) } \mathrm { L } _ { \mathrm { i } } ( \hat { \omega } ) ( \hat { \omega } \cdot \hat { n } ) e ^ { \mathrm { j } k \hat { \omega } _ { x y } \cdot \vec { \rho } } \mathrm { d } \sigma ( \hat { \omega } )\tag{35}
$$

$$
\mathrm { L } _ { \mathrm { i } } ( \boldsymbol { \hat { \omega } } ) = \frac { k ^ { 2 } } { 4 \pi ^ { 2 } } \int _ { \mathbb { R } ^ { 2 } } \boldsymbol { A } _ { \mathrm { i } } ( \vec { \rho } ; k ) e ^ { - \mathrm { j } k \hat { \omega } _ { x y } \cdot \vec { \rho } } \mathrm { d } \vec { \rho } , \quad \boldsymbol { \hat { \omega } } \in \mathcal { H } ^ { 2 } ( \boldsymbol { \hat { n } } )\tag{36}
$$

It is easily shown using the results of the previous section that the coherence function represents the two-point correlation of the incident field $\operatorname { U } _ { \mathrm { i } } ( \cdot , k )$ , where the Fourier variable $\vec { \rho }$ represents the separation between these two points <sup>8</sup>

$$
\begin{array} { r } { \varLambda _ { \mathrm { i } } ( \vec { \rho } _ { 1 } - \vec { \rho } _ { 2 } , k ) = \langle \mathrm { U } _ { \mathrm { i } } ( \vec { \rho } _ { 1 } , k ) \mathrm { U } _ { \mathrm { i } } ^ { * } ( \vec { \rho } _ { 2 } , k ) \rangle } \end{array}\tag{37}
$$

Observe that by definition, $\varLambda _ { \mathrm { i } } ( \overrightarrow { 0 } , k ) = \mathrm { E } _ { \mathrm { i } } ( k ) / \ell _ { \mathrm { i } } ( k )$

Example: Gaussian environment lighting. As a concrete example, consider the following lighting configuration<sup>9</sup>

$$
\mathrm { L } _ { \mathrm { i } } ( \hat { \omega } ) = \mathrm { L } _ { \mathrm { p } } \exp \left( - \frac { \left\| \hat { \omega } _ { x y } - \hat { \Omega } _ { x y } \right\| ^ { 2 } } { 2 \varDelta _ { \omega } ^ { 2 } } \right)\tag{38}
$$

The angular distribution of radiance is a Gaussian lobe (when projected onto the x-y plane) centered at $\hat { \varOmega } _ { x y }$ with (projected) angular spread $2 \varDelta _ { \omega }$ and peak value $\mathrm { L } _ { \mathrm { p } }$

The corresponding coherence function (Fourier dual of lighting) according to Definition 8 is given by

$$
A _ { \mathrm { i } } ( \vec { \rho } ; k ) = 2 \pi \varDelta _ { \omega } ^ { 2 } \mathrm { L } _ { \mathrm { p } } \exp \left( - \frac { \varDelta _ { \omega } ^ { 2 } k ^ { 2 } } { 2 } \| \vec { \rho } \| ^ { 2 } \right) \exp \left( \mathrm { j } k \hat { \varOmega } _ { x y } \cdot \vec { \rho } \right)\tag{39}
$$

$$
= \frac { \mathrm { E } _ { \mathrm { i } } ( k ) } { \ell _ { \mathrm { i } } ( k ) } \exp \left( - \frac { \| \vec { \rho } \| ^ { 2 } } { 2 \varDelta _ { \mathrm { i } } ^ { 2 } } \right) \exp \left( \mathrm { j } k \hat { \varOmega } _ { x y } \cdot \vec { \rho } \right)\tag{40}
$$

where $\varDelta _ { \mathrm { i } } = { 1 } / { \varepsilon \varDelta _ { \omega } }$ is the spatial coherence length, a measure of how quickly the magnitude of field correlation (coherence) decreases with separation, and $\mathrm { E } _ { \mathrm { i } } ( k ) = 2 \pi \varDelta _ { \omega } ^ { 2 } \mathrm { L } _ { \mathrm { p } } \ell _ { \mathrm { i } } ( k )$ is the spectral density of incident irradiance at the surface. We will use the above example in § F to prove Proposition 6 from the main text.

## E.3 Scattering and Measurement

Our goal now is to express the irradiance measured at a point on the camera sensor given the incident environment map $\mathrm { L } _ { \mathrm { i } } ( \hat { \omega } ) \ell _ { \mathrm { i } } ( \boldsymbol { k } )$ and the surface heightfield $\mathrm { h } ( \cdot )$ , which together (along with the camera lens) determine the scattered field reaching the camera sensor. For a given point on the camera sensor, consider $\overrightarrow { p }$ as the point on the tangent plane it is mapped to via pinhole projection (center of the lens aperture). The viewing angle of the camera for point $\overrightarrow { p }$ is vˆ (refer Figure 3).

Roughness model. We characterize the roughness of the random heightfield h using the standard deviation $\sigma _ { \mathrm { r } } : = \sqrt { \mathrm { v a r } ( \mathrm { h } ( \vec { x } ) ) }$ and correlation length $\varDelta _ { \mathrm { r } }$ $( \mathrm { c o r r } ( \mathrm { h } ( \vec { x } ) , \mathrm { h } ( \vec { y } ) ) \approx 0 \mathrm { i f } \| \vec { x } - \vec { y } \| > \Delta _ { \mathrm { r } } )$ . We consider 1. $\sigma _ { \mathrm { r } }$ comparable to wavelength, 2. ∆<sub>r</sub> smaller than the spatial coherence length $\varDelta _ { \mathrm { i } }$ . The same conditions underlie standard BRDF models for optically rough surfaces [64–66]. Thus in practice, they are satisfied by any surface not optically smooth (i.e., near-specular) at visible wavelengths. The surface roughness conditions are also spelt out in Goodman [46, §5.10]

Scattering integral. For a single incident plane wave $\mathcal { G } ( \hat { \omega } , \boldsymbol { k } ) e ^ { \mathrm { j } \boldsymbol { k } \hat { \omega } \cdot \mathbf { r } }$ , the scattered wave reaching the camera sensor is given by

$$
\mathrm { U _ { s } } ( \vec { p } ; k , \hat { \omega } ) = \mathcal { G } ( \hat { \omega } , k ) \cdot \widetilde { \mathrm { F } } ( \vec { p } ; k ( \hat { v } + \hat { \omega } ) )\tag{41}
$$

where we have defined the scattering integral [46, 73]

$$
\widetilde { \mathrm { F } } ( \overrightarrow { p } ; k , \overrightarrow { u } ) = \int _ { \mathbb { R } ^ { 2 } } \mathrm { K } _ { \mathrm { c } } ( \overrightarrow { p } - \overrightarrow { \rho } ) \mathrm { s } ( \overrightarrow { \rho } ; k , u _ { z } ) e ^ { \mathrm { j } k \overrightarrow { u } _ { x y } \cdot \overrightarrow { \rho } } \mathrm { d } \overrightarrow { \rho }\tag{42}
$$

where $\operatorname { K } _ { \mathrm { c } } ( \cdot )$ is the coherent spread function (CSF) of the camera describing the complex field impulse response of the camera lens at the depth of the surface (assumed fronto-parallel to the camera), and $\begin{array} { r } { \mathrm { s } ( \vec { \rho } ; k , u _ { z } ) = \bar { \mathrm { r } ( \vec { \rho } , k ) } e ^ { \mathrm { j } k u _ { z } \mathrm { h } ( \vec { \rho } ) } } \end{array}$ as introduced in Definition 3. The reflection coeficient $\operatorname { r } ( { \vec { \rho } } , k )$ is a slowly varying (compared to h) non-negative factor that models the spatially-varying spectral albedo of the surface. Note that $\mathrm { s } ( \cdot ; k , u _ { z } )$ is a random signal through its dependence on the heightfield h( ). This therefore results in the total field

$$
\mathrm { U _ { s } } ( \vec { p } ) = \int _ { \mathbb { R } ^ { + } } \int _ { \mathcal { H } ^ { 2 } ( \hat { n } ) } \mathcal { G } ( \hat { \omega } , k ) \cdot \widetilde { \mathrm { F } } ( \vec { p } ; k ( \hat { v } + \hat { \omega } ) ) \mathrm { d } \sigma ( \hat { \omega } ) \mathrm { d } k\tag{43}
$$

Measured irradiance. The irradiance reaching the sensor is the average meansquared of the field reaching the sensor, i.e.

$$
\begin{array} { r l r } {  { \mathrm { E } _ { \mathrm { s } } ( \vec { p } ) = \Big \langle | \mathrm { U } _ { \mathrm { s } } ( \vec { p } ) | ^ { 2 } \Big \rangle } } \\ & { } & { = \displaystyle \int _ { k \in \mathbb { R } ^ { + } } \int _ { \hat { \omega } \in \mathcal { H } ^ { 2 } ( \hat { n } ) } \ell _ { \mathrm { i } } ( k ) \mathrm { L } _ { \mathrm { i } } ( \hat { \omega } ) ( \hat { \omega } \cdot \hat { n } ) \Big | \widetilde { \mathrm { F } } ( \vec { p } ; k , ( \hat { v } + \hat { \omega } ) ) \Big | ^ { 2 } \mathrm { d } \sigma ( \hat { \omega } ) \mathrm { d } k } \end{array}\tag{44}
$$

(45)

following a simplification similar to the example of incident irradiance in $\ S$ E.1 based on the mutual incoherence assumption.

To obtain the irradiance measured by the camera sensor, we must weight the spectral integration by the sensor spectral sensitivity function $\mathrm { w } _ { \mathrm { c } } ( { k } )$

$$
\operatorname { E } ( { \overrightarrow { p } } ) = \int _ { \mathbb { R } ^ { + } } \operatorname { w } _ { \operatorname { c } } ( k ) \ell _ { \mathrm { i } } ( k ) \operatorname { E } ( { \overrightarrow { p } } , k ) \operatorname { d } k\tag{46}
$$

defining

$$
\mathrm { E } ( \vec { p } , k ) = \int _ { \mathcal { H } ^ { 2 } ( \hat { n } ) } \mathrm { L } _ { \mathrm { i } } ( \hat { \omega } ) ( \hat { \omega } \cdot \hat { n } ) \Big | \widetilde { \mathrm { F } } ( \vec { p } ; k , \hat { v } + \hat { \omega } ) \Big | ^ { 2 } \mathrm { d } \sigma ( \hat { \omega } )\tag{47}
$$

We therefore have the following proposition:

## Proposition 9: Measured Sensor Irradiance (Exact)

$$
\operatorname { E } ( { \overrightarrow { p } } ) = \int _ { \mathbb { R } ^ { + } } \operatorname { w } _ { \operatorname { c } } ( k ) \ell _ { \mathrm { i } } ( k ) \operatorname { E } ( { \overrightarrow { p } } , k ) \operatorname { d } k\tag{48}
$$

$$
\mathrm { E } ( \vec { p } , k ) = \int _ { \mathcal { H } ^ { 2 } ( \hat { n } ) } \mathrm { L } _ { \mathrm { i } } ( \hat { \omega } ) ( \hat { \omega } \cdot \hat { n } ) \Big | \widetilde { \mathrm { F } } ( \vec { p } ; k , \hat { v } + \hat { \omega } ) \Big | ^ { 2 } \mathrm { d } \sigma ( \hat { \omega } )\tag{49}
$$

## E.4 Approximating measured irradiance

To arrive at Proposition 4 in the form of a simple linear system, we now make two simplifying approximations.

Approximation 1: Mean z-component approximation. We now make a standard implicit approximation used in similar derivations [65, 66]. Consider the scattering integral Equation (42). The phase term $\mathrm { s } ( \vec { \rho } ; k \dot { u _ { z } } ) = e ^ { \dot { \mathrm { j } } k u _ { z } \mathrm { h } ( \vec { \rho } ) }$ varies with incident direction $\hat { \omega } .$ , since $\vec { u } = \hat { \omega } + \hat { v }$ in Equation (49), preventing an interpretation of $\widetilde { \mathrm { F } }$ as a Fourier transform.

eIf we consider an illumination environment map similar to Equation (38) where the incident illumination is relatively concentrated around a peak direction $\hat { \varOmega }$ with z-component $\varOmega _ { z }$ , then the variation in the z-component of the incident direction $\omega _ { z }$ across the environment map is small, $i . e . \ \omega _ { z } \approx \varOmega _ { z }$ for all $\hat { \omega }$ with significant contribution. Formally, the approximation is stated as:

$$
( \mathrm { h } ( \vec { \rho } _ { 1 } ) - \mathrm { h } ( \vec { \rho } _ { 2 } ) ) { \varDelta } _ { \omega _ { z } } \ll \lambda \quad \forall \vec { \rho } _ { 1 } , \vec { \rho } _ { 2 }\tag{50}
$$

This gives us the following approximate form for the scattering integral:

$$
\widetilde { \mathrm { F } } ( \overrightarrow { p } ; k \overrightarrow { u } ) \approx \int _ { \mathbb { R } ^ { 2 } } \mathrm { K } _ { \mathrm { c } } ( \overrightarrow { p } - \overrightarrow { \rho } ) \mathrm { s } ( \overrightarrow { \rho } ; k , \overrightarrow { u } _ { z } ) e ^ { \mathrm { j } k \overrightarrow { u } _ { x y } \cdot \overrightarrow { \rho } } \mathrm { d } \overrightarrow { \rho }\tag{51}
$$

using s $( \vec { \rho } ; k , \overline { { u } } _ { z } ) = e ^ { \mathrm { j } k \overline { { u } } _ { z } \mathrm { h } ( \vec { \rho } ) }$ with the mean direction z-component $\overline { { u } } _ { z } = \varOmega _ { z } + v _ { z }$

Using this approximate form, we expand the squared magnitude term in Equation (49) as

$$
\left| \widetilde { \mathrm { F } } ( \vec { p } ; k \vec { u } ) \right| ^ { 2 } \approx \int _ { \mathbb { R } ^ { 2 } } \int _ { \mathbb { R } ^ { 2 } } \mathrm { K } _ { \mathrm { c } } ( \vec { p } - \vec { \rho } _ { 1 } ) \mathrm { K } _ { \mathrm { c } } ^ { * } ( \vec { p } - \vec { \rho } _ { 2 } ) \mathrm { s } ( \vec { \rho } _ { 1 } ; k , \overline { { u } } _ { z } ) \mathrm { s } ^ { * } ( \vec { \rho } _ { 2 } ; k , \overline { { u } } _ { z } )\tag{52}
$$

Using a change of variables

$$
\begin{array} { c } { { \vec { q } = \displaystyle \frac { \vec { \rho } _ { 1 } + \vec { \rho } _ { 2 } } { 2 } } } \\ { { \vec { r } = \displaystyle \frac { \vec { \rho } _ { 1 } - \vec { \rho } _ { 2 } } { 2 } } } \end{array}\tag{53}
$$

we have

$$
\begin{array} { r l } & { \left| \widetilde { \mathrm { F } } ( \vec { p } ; k \vec { u } ) \right| ^ { 2 } \approx 4 \displaystyle \int _ { \mathbb { R } ^ { 2 } } \int _ { \mathbb { R } ^ { 2 } } \mathrm { K } _ { \mathrm { c } } ( \vec { p } - \vec { q } - \vec { r } ) \mathrm { K } _ { \mathrm { c } } ^ { * } ( \vec { p } - \vec { q } + \vec { r } ) } \\ & { \quad \quad \quad \mathrm { s } ( \vec { q } + \vec { r } ; k , \overline { { u } } _ { z } ) \mathrm { s } ^ { * } ( \vec { q } - \vec { r } ; k , \overline { { u } } _ { z } ) } \\ & { \quad \quad \quad \quad \quad e ^ { \mathrm { j } k \vec { u } _ { x y } \cdot ( 2 \vec { r } ) } \mathrm { d } \vec { r } \mathrm { d } \vec { q } } \end{array}\tag{54}
$$

Using $S ( \vec { q } , \vec { r } ; k , \overline { { u } } _ { z } )$ from Definition 3,

$$
S ( \vec { q } , \vec { r } ; k , \overline { { u } } _ { z } ) : = \mathbf { s } ( \vec { q } + \vec { r } ; k , \overline { { u } } _ { z } ) \mathbf { s } ^ { * } ( \vec { q } - \vec { r } ; k , \overline { { u } } _ { z } )\tag{55}
$$

and plugging the above into Equation (49), we have an approximate expression

$$
\begin{array} { l } { \displaystyle \mathrm { E } \big ( \overrightarrow { p } , k \big ) \approx 4 \int _ { \mathbb { R } ^ { 2 } } \int _ { \mathbb { R } ^ { 2 } } \mathrm { K } _ { \mathrm { c } } \big ( \overrightarrow { p } - \overrightarrow { q } - \overrightarrow { r } \big ) \mathrm { K } _ { \mathrm { c } } ^ { * } \big ( \overrightarrow { p } - \overrightarrow { q } + \overrightarrow { r } \big ) } \\ { \displaystyle S \big ( \overrightarrow { q } , \overrightarrow { r } ; k \overrightarrow { u } _ { z } \big ) } \\ { \displaystyle \Big [ \int _ { \mathcal { H } ^ { 2 } ( \hat { n } ) } \mathrm { L } _ { \mathrm { i } } \big ( \hat { \omega } \big ) \big ( \hat { \omega } \cdot \hat { n } \big ) e ^ { \mathrm { j } k \hat { \omega } _ { x y } \cdot \big ( 2 \overrightarrow { r } \big ) } \mathrm { d } \sigma ( \hat { \omega } ) \Big ] } \\ { \displaystyle e ^ { \mathrm { j } k \hat { v } _ { x y } \cdot \big ( 2 \overrightarrow { r } \big ) } \mathrm { d } \overrightarrow { r } \mathrm { d } \overrightarrow { q } } \end{array}\tag{56}
$$

Identifying the term in square brackets as the coherence function $\varLambda _ { \mathrm { i } } ( 2 \overrightarrow { r } ; k )$ from Definition 8, we have

$$
\begin{array} { r } { \mathrm { E } ( \overrightarrow { p } , k ) \approx 4 \displaystyle \int _ { \mathbb { R } ^ { 2 } } \int _ { \mathbb { R } ^ { 2 } } \mathrm { K } _ { \mathrm { c } } ( \overrightarrow { p } - \overrightarrow { q } - \overrightarrow { r } ) \mathrm { K } _ { \mathrm { c } } ^ { \ast } ( \overrightarrow { p } - \overrightarrow { q } + \overrightarrow { r } ) } \\ { S ( \overrightarrow { q } , \overrightarrow { r } ; k \overrightarrow { u } _ { z } ) \varLambda _ { \mathrm { i } } ( 2 \overrightarrow { r } ; k ) e ^ { \mathrm { j } k \hat { v } _ { x y } \cdot ( 2 \overrightarrow { r } ) } \mathrm { d } \overrightarrow { r } \mathrm { d } \overrightarrow { q } } \end{array}\tag{57}
$$

At this point, we bring to attention the definitions of $\mathrm { L } _ { \mathrm { o } }$ from Definition 3

$$
\mathrm { L } _ { \mathrm { o } } \bigl ( \vec { q } , \hat { v } , k \bigr ) : = \int _ { \mathcal { H } ^ { 2 } ( \hat { n } ) } \mathrm { f } _ { \mathrm { r } } ^ { \mathrm { h } } \bigl ( \vec { q } , \hat { \omega } + \hat { v } , k \bigr ) \mathrm { L } _ { \mathrm { i } } ( \hat { \omega } ) \bigl ( \hat { \omega } \cdot \hat { n } \bigr ) \mathrm { d } \sigma ( \hat { \omega } ) .\tag{rep. 7}
$$

and Definition 3

$$
\mathrm { f } _ { \mathrm { r } } ^ { \mathrm { h } } \big ( \vec { q } , \vec { u } , k \big ) : = \frac { 1 } { A } \int _ { \mathbb { R } ^ { 2 } } S \big ( \vec { q } , \vec { r } , k , u _ { z } \big ) e ^ { \mathrm { j } k \vec { u } _ { x y } \cdot 2 \vec { r } } \mathrm { d } \vec { r } ,\tag{rep. 4}
$$

where $A ,$ with dimensions of area, is a constant inversely proportional to the amount of light collection by the lens aperture, scaling as $A \propto N _ { \mathrm { e f f } } ^ { 2 } / 4 m ^ { 2 }$ (the proportionality absorbs the area scale set by the CSF, which cancels in all downstream results). The efective (sensor-side) f-number $N _ { \mathrm { e f f } } = N ( 1 + m )$ , N being the f-number of the camera (refer $\ S \ O \mathrm { ~ A ~ } \ O$

With the definition of the coherence function in Definition $^ { 8 , }$ and using the same mean-z-component approximation already used for the scattering integral ${ \widetilde { \mathrm { F } } } ,$ we can write $\mathrm { L } _ { \mathrm { o } } \big ( \overrightarrow { q } , \hat { v } , \boldsymbol { k } \big )$ approximately as

$$
\mathrm { L } _ { \mathrm { o } } ( \vec { q } , \hat { v } , k ) \approx \frac { 1 } { A } \int _ { \mathbb { R } ^ { 2 } } S ( \vec { q } , \vec { r } ; k , \overline { { u } } _ { z } ) \varLambda _ { \mathrm { i } } ( 2 \vec { r } ; k ) e ^ { \mathrm { j } k \hat { v } _ { x y } \cdot ( 2 \vec { r } ) } \mathrm { d } \vec { r }\tag{58}
$$

Approximation 2: Weak-coherence approximation. For typical ambientlighting conditions, the coherence function $( e . g$ . Equation (39)) will have small width $\varDelta _ { \mathrm { i } } = { 1 } / { k } { \varDelta _ { \omega } }$ . As long as $\varDelta _ { \mathrm { i } }$ is small relative to the width of the CSF $\operatorname { K } _ { \mathrm { c } } ( \cdot )$ 2 we can approximate the CSF to be constant within the area of the support of $\varLambda _ { \mathrm { i } } ,$ resulting in the expression (from Equation (57) and Equation (58)):

$$
\mathrm { E } ( \vec { p } , k ) \approx \int _ { \mathbb { R } ^ { 2 } } \mathrm { P } _ { \mathrm { c } } ( \vec { p } - \vec { q } ) \mathrm { L } _ { \mathrm { o } } ( \vec { q } , \hat { v } , k ) \mathrm { d } \vec { q }\tag{59}
$$

where the normalized PSF (integrates to unity) of the camera $\mathrm { P _ { c } } ( \overrightarrow { \rho } ) : = { \cal N } _ { \mathrm { e f f } } ^ { 2 } / m ^ { 2 } \big | \mathrm { K _ { c } } ( \overrightarrow { \rho } ) \big | ^ { 2 }$ Combining the above with Equation (48), we have

$$
\mathrm { E } ( \vec { p } ) \approx \int _ { \mathbb { R } ^ { + } } \mathrm { w } _ { \mathrm { c } } ( k ) \ell _ { \mathrm { i } } ( k ) \int _ { \mathbb { R } ^ { 2 } } \mathrm { P } _ { \mathrm { c } } ( \vec { p } - \vec { q } ) \mathrm { L } _ { \mathrm { o } } ( \vec { q } , \hat { v } , k ) \mathrm { d } \vec { q } \mathrm { d } k\tag{60}
$$

thus proving Proposition 4.

## E.5 Ensemble averaging

The BRDF defined in Stam [65] can be equivalently stated using our notation as the (normalized) power spectral density of the scattering function s:

$$
\overline { { \mathrm { f } } } _ { \mathrm { r } } ( \overrightarrow { q } , \overrightarrow { u } , k ) = \frac { 1 } { 4 A } \mathcal { P } _ { \mathrm { s } } ( k \overrightarrow { u } _ { x y } ) ,\tag{61}
$$

$$
\mathcal { P } _ { \mathrm { s } } ( \vec { \kappa } ) : = \operatorname* { l i m } _ { A  \infty } \frac { 1 } { A } \mathbb { E } _ { \mathrm { h } } [ | \int _ { A } \mathrm { s } ( \vec { \rho } ; k , u _ { z } ) e ^ { \mathrm { j } \vec { \kappa } \cdot \vec { \rho } } \mathrm { d } \vec { \rho } | ^ { 2 } ] ,\tag{62}
$$

where $\mathcal { A }$ is the illuminated surface area and $\mathcal { P } _ { \mathrm { s } }$ is the power spectral density of s (the Fourier transform of its spatial autocorrelation, by the Wiener–Khinchin theorem). The $^ { 1 / \mathcal { A } }$ normalization makes the BRDF independent of the illuminated area (a per-unit-area surface property). For this section, we are considering a surface where the BRDF is constant (as indicated by the expression). The BRDF is proportional to the expected value of the squared magnitude of the absolutesquared fourier transform of $\mathrm { s } ( \vec { \rho } ; k , u _ { z } )$ over an ensemble of surface heightfield realizations. It is almost the mean of the absolute-square of the scattering integral F in Equation (42), except for the camera lens $\mathrm { C S F }$ term $\operatorname { K } _ { \mathrm { c } } ( \cdot )$

The expression for irradiance in our model depends on the particular realization of the surface heightfield $\mathrm { h } ( \cdot )$ through the sample $B R D F ~ \mathrm { f _ { r } ^ { h } }$ . We first prove Proposition 5—that the ensemble mean of the sample BRDF is the traditional BRDF—and then use the exact model to show how the finite-aperture CSF relates the two.

Proof of Proposition 5. If we compute the mean of our sample BRDF $\mathrm { f } _ { \mathrm { r } } ^ { \mathrm { h } } ( \overline { { q } } , \overline { { u } } , k )$ from Definition 3 over an ensemble of surface heightfield realizations, we have

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { \mathrm { h } } \big [ \mathrm { f } _ { \mathrm { r } } ^ { \mathrm { h } } ( \vec { q } , \vec { u } , k ) \big ] = \frac { 1 } { A } \int _ { \mathbb { R } ^ { 2 } } \mathbb { E } _ { \mathrm { h } } [ S ( \vec { q } , \vec { r } , k , u _ { z } ) ] e ^ { \mathrm { j } k \vec { u } _ { x y } \cdot 2 \vec { r } } \mathrm { d } \vec { r } } \\ { \displaystyle \qquad = \frac { 1 } { A } \int _ { \mathbb { R } ^ { 2 } } \mathbb { E } _ { \mathrm { h } } \big [ \mathrm { s } ( \vec { q } + \vec { r } ; k , u _ { z } ) \mathrm { s } ^ { * } ( \vec { q } - \vec { r } ; k , u _ { z } ) \big ] e ^ { \mathrm { j } k \vec { u } _ { x y } \cdot 2 \vec { r } } \mathrm { d } \vec { r } } \end{array}\tag{63}
$$

(64)

This step requires only wide-sense stationarity of s (equivalently, of the heightfield h): the correlation $\mathbb { E } _ { \mathrm { h } } [ \mathrm { s } ( \vec { q } + \vec { r } ) \mathrm { s } ^ { * } ( \vec { q } - \vec { r } ) ]$ is then the autocorrelation of s at separation $2 \vec { r }$ , independent of the center $\dot { \vec { q } } . ^ { 1 0 }$ Substituting ${ \vec { \rho } } = 2 { \vec { r } }$ (Jacobian $\mathrm { d } \overrightarrow { r } = 1 / 4 \mathrm { ~ d } \overrightarrow { \rho } )$

$$
\mathbb { E } _ { \mathrm { h } } \big [ \mathrm { f } _ { \mathrm { r } } ^ { \mathrm { h } } ( \vec { q } , \vec { u } , k ) \big ] = \frac { 1 } { 4 A } \int _ { \mathbb { R } ^ { 2 } } \mathbb { E } _ { \mathrm { h } } \big [ \mathrm { s } ( \vec { \rho } ^ { \prime } + \vec { \rho } ; k , u _ { z } ) \mathrm { s } ^ { * } ( \vec { \rho } ^ { \prime } ; k , u _ { z } ) \big ] e ^ { \mathrm { j } k \vec { u } _ { x y } \cdot \vec { \rho } } \mathrm { d } \vec { \rho }\tag{65}
$$

(66)

identifying the Fourier transform of the autocorrelation with the power spectral density $\mathcal { P } _ { \mathrm { s } }$ (Wiener–Khinchin). Comparing with the definition of ${ \overline { { \mathrm { f } } } } _ { \mathrm { r } } .$ we obtain (Proposition 5)

$$
\overline { { \mathrm { f } } } _ { \mathrm { r } } ( \vec { q } , \hat { \omega } + \hat { v } , \boldsymbol { k } ) = \mathbb { E } _ { \mathrm { h } } \big [ \mathrm { f } _ { \mathrm { r } } ^ { \mathrm { h } } ( \vec { q } , \hat { \omega } + \hat { v } , \boldsymbol { k } ) \big ] .\tag{67}
$$

The traditional BRDF as the pinhole limit. If we consider the expected value of the irradiance in Equation (49) from the exact model (Proposition 9), we obtain

$$
\mathbb { E } _ { \mathrm { h } } [ \mathrm { E } ( \vec { p } , k ) ] = \int _ { \mathcal { H } ^ { 2 } ( \hat { n } ) } \mathrm { L } _ { \mathrm { i } } ( \hat { \omega } ) ( \hat { \omega } \cdot \hat { n } ) \mathbb { E } _ { \mathrm { h } } \bigg [ \Big | \widetilde { \mathrm { F } } ( \vec { p } ; k , \hat { v } + \hat { \omega } ) \Big | ^ { 2 } \bigg ] \mathrm { d } \sigma ( \hat { \omega } )\tag{68}
$$

$$
\mathbb { E } _ { \mathrm { h } } \bigg [ \bigg | \widetilde { \mathrm { F } } ( \overrightarrow { p } ; k , \hat { v } + \hat { \omega } ) \bigg | ^ { 2 } \bigg ] = \mathbb { E } _ { \mathrm { h } } \bigg [ \bigg | \int _ { \mathbb { R } ^ { 2 } } \mathrm { K } _ { \mathrm { c } } \big ( \overrightarrow { p } - \overrightarrow { \rho } \big ) \mathrm { s } ( \overrightarrow { \rho } ; k , u _ { z } ) e ^ { \mathrm { j } k \overrightarrow { u } _ { x y } \cdot \overrightarrow { \rho } } \mathrm { d } \overrightarrow { \rho } \bigg | ^ { 2 } \bigg ]\tag{69}
$$

If we consider the case of a pinhole camera, then $\mathrm { K } _ { \mathrm { c } } ( \cdot )$ has infinite extent and is constant across the surface, so it factors out of ${ \widetilde { \mathrm { F } } } _ { \cdot }$ leaving the plain Fourier transform of s that defines the traditional BRDF:

$$
\mathbb { E } _ { \mathrm { h } } \bigg [ \Big | \widetilde { \mathrm { F } } ( \vec { p } ; k , \hat { v } + \hat { \omega } ) \Big | ^ { 2 } \bigg ] \propto \mathbb { E } _ { \mathrm { h } } \bigg [ \bigg | \int _ { \mathcal { A } } \mathrm { s } ( \vec { \rho } ; k , u _ { z } ) e ^ { \mathrm { i } k \vec { u } _ { x y } \cdot \vec { \rho } } \mathrm { d } \vec { \rho } \bigg | ^ { 2 } \bigg ] \propto \mathcal { P } _ { \mathrm { s } } ( k \vec { u } _ { x y } ) \propto \overline { { \mathrm { f } } } _ { \mathrm { r } } ( \vec { q } , \vec { u } , k ) ,\tag{70}
$$

recovering the traditional BRDF-based rendering equation. The two agree only in this pinhole limit: the traditional BRDF omits the camera’s coherent spread function $\mathrm { K _ { c } }$ , whereas our exact model retains it. For any finite aperture the CSF remains inside $\widetilde { \mathrm { F } }$ and our model departs from the traditional BRDF—precisely ethe regime in which defocus and second-order texture (subjective speckle) arise. Our model thus generalizes the traditional BRDF to finite-aperture imaging of unresolved microgeometry.

## F Characterization of second-order texture

In this section, we use the appearance model derived in the previous section to prove Proposition 6 in the main text, characterizing the distribution of irradiance through $\kappa _ { \mathrm { E } } ^ { 2 }$

Equation (59) can be approximated as:

$$
\mathrm { E } ( \vec { p } , k ) \approx \sum _ { m } \mathrm { E } _ { m } ^ { ( \epsilon ) } , \quad \mathrm { w h e r e }\tag{71}
$$

$$
\mathrm { E } _ { m } ^ { ( \epsilon ) } ( k ) : = \epsilon ^ { 2 } \mathrm { P } _ { \mathrm { c } } ( \overrightarrow { p } - \overrightarrow { q } _ { m } ) \mathcal { L } ^ { ( \epsilon ) } ( \overrightarrow { q } _ { m } , k ) ,\tag{72}
$$

$$
\mathcal { L } ^ { ( \epsilon ) } ( \vec { q } , k ) : = \int _ { \mathbb { R } ^ { 2 } } G _ { \epsilon } \bigl ( \vec { q } ^ { \prime } - \vec { q } \bigr ) \mathrm { L } _ { \circ } \bigl ( \vec { q } ^ { \prime } , \hat { v } , k \bigr ) \mathrm { d } \vec { q } ^ { \prime }\tag{73}
$$

defining $\mathcal { L } ^ { ( \epsilon ) } ( \cdot , k )$ as the (blurred) local average of outgoing radiance $\operatorname { L } _ { \mathrm { o } } ( \cdot , \hat { v } , k )$ using a normalized Gaussian kernel in $\mathbb { R } ^ { 2 }$ of width ϵ centered at $\overrightarrow { q } _ { r }$ m

$$
G _ { \epsilon } ( \overrightarrow { q } ^ { \prime } - \overrightarrow { q } _ { m } ) : = \frac { 2 } { \pi \epsilon ^ { 2 } } \exp { \left( - 2 \frac { \| \overrightarrow { q } ^ { \prime } - \overrightarrow { q } _ { m } \| ^ { 2 } } { \epsilon ^ { 2 } } \right) } ,\tag{74}
$$

with $\left\{ \overrightarrow { q } _ { m } \right\}$ forming a rectangular grid of spacing ϵ in $\mathbb { R } ^ { 2 }$ . In the limit of $\epsilon  0$ Equation (71) becomes an exact equality with Equation (59). This approximation is suitable when ϵ is smaller than the width of the PSF $\varDelta _ { \mathrm { c } }$

Our objective now is to model the distribution of $\operatorname { E } ( { \overrightarrow { p } } , k )$ . To do this, we consider the approximation above using the specific grid spacing of $\epsilon = \varDelta _ { \mathrm { i } }$ . In § F.1, we show that considering the Gaussian environment lighting model in Equations (38)–(39) with spatial coherence length $\varDelta _ { \mathrm { i } } , \varDelta ^ { ( \varDelta _ { \mathrm { i } } ) } ( \vec { q } , k )$ can be modeled as an exponential random variable:

$$
\mathcal { L } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \vec { q } , k ) \mid \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \vec { q } , k ) \sim \mathrm { E x p o } \left( 1 / \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \vec { q } , k ) \right) ,\tag{75}
$$

with mean equal to its first-order (ensemble-averaged) counterpart, $\overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \cdot , k ) =$ $\mathbb { E } _ { \mathrm { h } } \big [ \mathcal { L } ^ { ( \Delta _ { \mathrm { i } } ) } ( \cdot , k ) \big ]$ . More explicitly in terms of the (first-order) BRDF:

$$
\overline { { \mathcal { L } } } ^ { ( \mathcal { A } _ { \mathrm { i } } ) } ( \vec { q } , k ) = \int _ { \mathbb { R } ^ { 2 } } G _ { \epsilon } ( \vec { \rho } - \vec { q } ) \int _ { \mathcal { H } ^ { 2 } ( \hat { n } ) } \mathbb { E } _ { \mathrm { h } } \bigl [ \mathrm { f } _ { \mathrm { r } } ^ { \mathrm { h } } ( \vec { \rho } , \hat { \omega } , \hat { \nu } , k ) \bigr ] \mathrm { L } _ { \mathrm { i } } ( \hat { \omega } ) ( \hat { \omega } \cdot \hat { n } ) \mathrm { d } \sigma ( \hat { \omega } ) \mathrm { d } \vec { \rho }\tag{76}
$$

We can also consider $\mathcal { L } ^ { ( \Delta _ { \mathrm { i } } ) } ( \vec { q } _ { i } , k )$ and $\mathcal { L } ^ { ( \Delta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { j } , k )$ are independent for $i \neq j$ since they correspond to (mostly) non-overlapping microstructure regions on the surface (and the correlation length of the heightfield is smaller than $\varDelta _ { \mathrm { i } } )$ . We can now approximate Equation (6) as:

$$
\mathrm { E } ( { \overrightarrow { p } } ) \approx \sum _ { m } A _ { \mathrm { i } } ^ { 2 } \mathrm { P } _ { \mathrm { c } } ( { \overrightarrow { p } } - { \overrightarrow { q } } _ { m } ) { \mathcal { L } } ^ { ( { \varDelta } _ { \mathrm { i } } ) } ( { \overrightarrow { q } } _ { m } ) , \quad { \mathrm { w h e r e } }\tag{77}
$$

$$
\mathcal { L } ^ { ( \Delta _ { \mathrm { i } } ) } ( \vec { q } ) : = \int _ { \mathbb { R } ^ { + } } \mathcal { L } ^ { ( \Delta _ { \mathrm { i } } ) } ( \vec { q } , k ) \mathrm { w } _ { \mathrm { c } } ( k ) \ell _ { \mathrm { i } } ( k ) \mathrm { d } k\tag{78}
$$

While $\mathcal { L } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \vec { q } _ { m } , k )$ are independent<sup>11</sup> across the grid locations $\vec { q } _ { m }$ , they are not independent across wavenumbers k for a fixed m, i.e. $\mathcal { L } ^ { ( \varDelta _ { \mathrm { i } } ) } \bigl ( \overrightarrow { q } _ { m } , k _ { 1 } \bigr )$ and $\mathcal { L } ^ { ( \Delta _ { \mathrm { i } } ) } ( \vec { q } _ { m } , k _ { 2 } )$ are not independent as they are both directly dependent on the local surface structure in the neighborhood of ${ \vec { q } } _ { m } .$ . Considering two wavenumbers $\boldsymbol { k } _ { 1 } , k _ { 2 }$ they may be correlated or uncorrelated depending on the distribution from which the surface heightfield h is sampled.

Consider that the normalized spectral envelope of the light reaching the camera

$$
\mathrm { f _ { 0 } } ( k ) : = \frac { \mathrm { w _ { c } } ( k ) \ell _ { \mathrm { i } } ( k ) } { \int _ { \mathbb { R } ^ { + } } \mathrm { w _ { c } } ( k ^ { \prime } ) \ell _ { \mathrm { i } } ( k ^ { \prime } ) \mathrm { d } k ^ { \prime } }\tag{79}
$$

has a gaussian form

$$
\mathrm { f } _ { 0 } ( k ) = \frac { 1 } { \sqrt { 2 \pi } \varDelta _ { k } } \exp \left( - \frac { \left( k - \overline { { k } } \right) ^ { 2 } } { 2 \varDelta _ { k } ^ { 2 } } \right)\tag{80}
$$

with bandwidth $\varDelta _ { k }$ and mean wavenumber ${ \overline { { k } } } .$ Goodman [46] show that for a Gaussian-distributed heightfield, Gaussian correlation function, and a Gaussian spectral profile, the number of uncorrelated spectral buckets for a given wavenumber bandwidth $\varDelta _ { k }$ is given by

$$
M = \sqrt { 1 + 8 \pi ^ { 2 } \bigg ( \frac { \varDelta _ { k } } { \overline { { k } } } \bigg ) ^ { 2 } \bigg ( \frac { \sigma _ { \mathrm { r } } } { \overline { { \lambda } } } \bigg ) ^ { 2 } } \approx G _ { 1 } \varDelta _ { k }\tag{81}
$$

where $G _ { 1 } = 2 \sqrt { 2 } \pi ( { \sigma _ { \mathrm { r } } } / { \overline { { \lambda k } } } )$ , thereby allowing us to approximately model the integral over wavenumber in Equation (78) as a sum of M uncorrelated exponential random variables.

The conditional distribution of $\mathcal { L } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \overrightarrow { q } )$ can therefore be approximated as a Gamma distribution with shape parameter $M \mathbf { \cdot }$

$$
\mathcal { L } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } ) \mid \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } ) \sim \mathrm { G a m m a } \left( M , \frac { M } { \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } ) } \right) .\tag{82}
$$

where we similarly define

$$
\overline { { \mathcal { L } } } ^ { ( \Delta _ { \mathrm { i } } ) } ( \vec { q } ) : = \int _ { \mathbb { R } ^ { + } } \overline { { \mathcal { L } } } ^ { ( \Delta _ { \mathrm { i } } ) } ( \vec { q } , k ) \mathrm { w } _ { \mathrm { c } } ( k ) \ell _ { \mathrm { i } } ( k ) \mathrm { d } k\tag{83}
$$

## Proposition 10: Model for second-order texture

The irradiance at point $\overrightarrow { p }$ can be approximately modeled as

$$
\mathrm { E } ( \overrightarrow { p } ) \approx \sum _ { m } { \varDelta \mathrm { i } ^ { 2 } \mathrm { P } _ { \mathrm { c } } ( \overrightarrow { p } - \overrightarrow { q } _ { m } ) \mathcal { L } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } ) }\tag{84}
$$

where the (blurred) outgoing radiance at grid locations $\vec { q } _ { m }$ with spacing $\varDelta _ { \mathrm { i } }$ can be modeled as Gamma-distributed random variables

$$
\mathcal { L } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } ) \mid \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } ) \sim \mathrm { G a m m a } \left( M , \frac { M } { \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } ) } \right) .\tag{85}
$$

We correspondingly define a first-order counterpart of the approximate irradiance

$$
\overline { { \mathrm { E } } } ( \overrightarrow { p } ) \approx \sum _ { m } { \varDelta \mathrm { i } ^ { 2 } \mathrm { P } _ { \mathrm { c } } ( \overrightarrow { p } - \overrightarrow { q } _ { m } ) \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } ) }\tag{86}
$$

Using the above model, we show in § F.2 that $\kappa _ { \mathrm { E } } ^ { 2 }$ can be written as

$$
\kappa _ { \mathrm { E } } ^ { 2 } : = \frac { 1 } { M Q } \Big ( 1 + \kappa _ { \overline { { \mathcal { L } } } ^ { ( \Delta _ { \mathrm { i } } ) } } ^ { 2 } \Big ) + \kappa _ { \overline { { \mathrm { E } } } } ^ { 2 } .\tag{87}
$$

where the respective coeficients of variation for E and $\overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) }$

$$
\kappa _ { \mathrm { { E } } } ^ { 2 } : = \frac { \mathbb { V } \left[ \overline { { \mathrm { E } } } ( \overline { { p } } ) \right] } { \mu _ { E } ^ { 2 } } ,\tag{88}
$$

$$
\kappa _ { \overline { { \mathcal { L } } } ^ { ( \Delta _ { \mathrm { i } } ) } } ^ { 2 } : = \frac { \mathbb { V } \left[ \overline { { \mathcal { L } } } ^ { ( \Delta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } ) \right] } { \mathbb { E } \left[ \overline { { \mathcal { L } } } ^ { ( \Delta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } ) \right] ^ { 2 } } .\tag{89}
$$

If we further assume that first-order texture is independent across the grid locations $\vec { q } _ { m }$ (i.e. $\overline { { \operatorname { E } } } ( \vec { q } _ { m } )$ are independent across $\vec { q } _ { m } )$ , then we can write $\kappa _ { \overline { { \mathrm { E } } } } ^ { 2 } =$ $\frac { 1 } { Q } \kappa _ { \overline { { \mathcal { L } } } ^ { ( \Delta _ { \mathrm { i } } ) } } ^ { 2 }$ , resulting in the simplified expression

$$
\kappa _ { \mathrm { E } } ^ { 2 } = \frac { 1 } { M Q } \Big ( 1 + \kappa _ { \overline { { \mathcal { L } } } ^ { ( \Delta _ { \mathrm { i } } ) } } ^ { 2 } \Big ) + \frac { 1 } { Q } \kappa _ { \overline { { \mathcal { L } } } ^ { ( \Delta _ { \mathrm { i } } ) } } ^ { 2 } .\tag{90}
$$

$$
= \frac { 1 } { M Q } \big ( 1 + Q \kappa _ { \mathrm { E } } ^ { 2 } \big ) + \kappa _ { \mathrm { E } } ^ { 2 }\tag{91}
$$

thus proving Proposition 6.

## F.1 Proof of exponential distribution

Plugging Equation (58) into Equation (73), we have

$$
\begin{array} { r } { \mathcal { L } ^ { ( \epsilon ) } ( \vec { q } ^ { \prime } , \boldsymbol { k } ) = \displaystyle \frac { 1 } { A } \int _ { \mathbb { R } ^ { 2 } } \int _ { \mathbb { R } ^ { 2 } } G _ { \epsilon } ( \vec { q } - \vec { q } ^ { \prime } ) S ( \vec { q } , \vec { r } , \boldsymbol { k } ) A _ { \mathrm { i } } ( 2 \vec { r } , \boldsymbol { k } ) e ^ { \mathrm { j } \boldsymbol { k } \hat { v } _ { x y } \cdot ( 2 \vec { r } ) } \mathrm { d } \vec { r } \mathrm { d } \vec { q } , \mathrm { ~ ( } 9 } \\ { = \displaystyle \frac { 2 } { \pi A \epsilon ^ { 2 } } \int _ { \mathbb { R } ^ { 2 } } \int _ { \mathbb { R } ^ { 2 } } e ^ { - 2 \frac { \| \vec { q } - \vec { q } ^ { \prime } \| ^ { 2 } } { \epsilon ^ { 2 } } S ( \vec { q } , \vec { r } , \boldsymbol { k } ) A _ { \mathrm { i } } ( 2 \vec { r } , \boldsymbol { k } ) e ^ { \mathrm { j } \boldsymbol { k } \hat { v } _ { x y } \cdot ( 2 \vec { r } ) } \mathrm { d } \vec { r } \mathrm { d } \vec { q } } } \end{array}\tag{2}
$$

(93)

To proceed we consider the special case of the coherence function for Gaussian environment lighting (Equations (38)–(39)):

$$
\varLambda _ { \mathrm { i } } ( 2 \vec { r } , k ) = \mathrm { E } _ { \mathrm { i } } ( k ) \exp \left( - \frac { \left\| 2 \vec { r } \right\| ^ { 2 } } { 2 \varDelta _ { \mathrm { i } } ^ { 2 } } \right) \exp \left( \mathrm { j } k \hat { \Omega } _ { x y } \cdot 2 \vec { r } \right)\tag{94}
$$

Using a change of variables $\vec { t } _ { 1 } = \vec { q } + \vec { r } , \vec { t } _ { 2 } = \vec { q } - \vec { r }$ , we have

$$
\begin{array} { r l } & { \mathcal { L } ^ { ( \epsilon ) } ( \vec { q } ^ { \prime } , k ) = \mathrm { E } _ { \mathrm { i } } ( k ) \frac { 2 e ^ { - \frac { 2 \left\| \vec { q } ^ { \prime } \right\| ^ { 2 } } { \epsilon ^ { 2 } } } } { \pi A \epsilon ^ { 2 } } \cdot } \\ & { \qquad \int _ { \mathbb { R } ^ { 2 } } \int _ { \mathbb { R } ^ { 2 } } \left( \vec { t } _ { 1 } , k \right) e ^ { - \frac { \left\| \vec { \tau } _ { 1 } \right\| ^ { 2 } } { 2 } \left( \frac { 1 } { \epsilon ^ { 2 } } + \frac { 1 } { A _ { 1 } ^ { 2 } } \right) + \frac { 2 \vec { \tau } ^ { \prime } \cdot \vec { T } _ { 1 } } { \epsilon ^ { 2 } } } e ^ { \mathrm { i } k \left( \hat { \nu } _ { x y } + \hat { \mathcal { T } } _ { x y } \right) \cdot \vec { \tau } _ { 1 } } } \\ & { \qquad \mathrm { s } ^ { \ast } \left( \vec { t } _ { 2 } , k \right) e ^ { - \frac { \left\| \vec { \tau } _ { 2 } \right\| ^ { 2 } } { 2 } \left( \frac { 1 } { \epsilon ^ { 2 } } + \frac { 1 } { A _ { 1 } ^ { 2 } } \right) + \frac { 2 \vec { \tau } ^ { \prime } \cdot \vec { T } _ { 2 } } { \epsilon ^ { 2 } } } e ^ { - \mathrm { i } k \left( \hat { \nu } _ { x y } + \hat { \mathcal { T } } _ { x y } \right) \cdot \vec { \tau } _ { 2 } } } \\ & { \qquad \mathrm { e x p } \Bigg ( { - \vec { t } _ { 1 } \cdot \vec { t } _ { 2 } \left( \frac { 1 } { \epsilon ^ { 2 } } - \frac { 1 } { \varDelta _ { 1 } ^ { 2 } } \right) } \Bigg ) \mathrm { d } \vec { t } _ { 1 } \mathrm { d } \vec { t } _ { 2 } } \end{array}\tag{95}
$$

The integrand above is separable in $\stackrel { \triangledown } { \boldsymbol { t } _ { 1 } }$ and $\overrightarrow { t } _ { 2 }$ except for the cross-term in the last exponential above. Setting $\epsilon = \varDelta _ { \mathrm { i } }$ , the cross-term in the exponential vanishes, and the above integral then simplifies to

$$
\mathscr { L } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \vec { q } ^ { \prime } , k ) = \frac { 2 \mathrm { E _ { i } } ( k ) } { \pi A \varDelta _ { \mathrm { i } } ^ { 2 } } \left| \int _ { \mathbb { R } ^ { 2 } } \mathrm { s } ( \vec { t } , k ) e ^ { - \frac { \| \vec { \tau } - \vec { q } ^ { \prime } \| ^ { 2 } } { \varDelta _ { \mathrm { i } } ^ { 2 } } } e ^ { \mathrm { j } k \left( \hat { v } _ { x y } + \hat { \varOmega } _ { x y } \right) \cdot \vec { t } } \mathrm { d } \vec { t } \right| ^ { 2 }\tag{96}
$$

$$
= \mathrm { E i } ( k ) { \frac { 2 \pi \varDelta _ { \mathrm { i } } ^ { 2 } } { A } } { \left| \int _ { \mathbb { R } ^ { 2 } } G _ { \sqrt { 2 } { \varDelta _ { \mathrm { i } } } } { \left( { \vec { t } } - { \vec { q } } ^ { \prime } \right) } \mathrm { s } { \left( { \vec { t } } , k \right) } e ^ { \mathrm { j } k { \left( { \hat { v } } _ { x y } + { \hat { \varOmega } } _ { x y } \right) } \cdot { \vec { t } } } \mathrm { d } { \vec { t } } \right| } ^ { 2 }\tag{97}
$$

Without ensemble averaging over the distribution of surface heightfields, the above corresponds to speckle with a Gaussian window of width $\sqrt { 2 } \varDelta _ { \mathrm { i } }$ around $\overrightarrow { q } ^ { \prime }$ on the surface. Since it is a magnitude-squared of a sum of a large number of independent complex random variables (considering correlation length of the height field is suficiently small relative to $\varDelta _ { \mathrm { i } } )$ , we can argue (as in Goodman [46]) that the real and imaginary parts of the integral are therefore normally distributed random variables (by central limit theorem), thereby making the blurred spectral density of outgoing radiance $\mathcal { L } ^ { ( \Delta _ { \mathrm { i } } ) } ( \vec { q } ^ { \prime } , k )$ exponentially distributed.

## F.2 Spatial integration

From Equation (77) we have the conditional mean of irradiance given the firstorder blurred outgoing radiance $\overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \vec { p } )$ as:

$$
\mathbb { E } \Big [ \mathrm { E } ( \vec { p } ) \mid \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } \Big ] = \sum _ { m } { \varDelta _ { \mathrm { i } } ^ { 2 } \mathrm { P } _ { \mathrm { c } } ( \vec { p } - \vec { q } _ { m } ) \mathbb { E } \Big [ \mathcal { L } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \vec { q } _ { m } ) \mid \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \vec { q } _ { m } ) \Big ] }\tag{98}
$$

$$
= \sum _ { m } \varDelta _ { \mathrm { i } } ^ { 2 } \mathrm { P } _ { \mathrm { c } } ( \overrightarrow { p } - \overrightarrow { q } _ { m } ) \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } )\tag{99}
$$

$$
\approx \int _ { \mathbb { R } ^ { 2 } } \mathrm { P } _ { \mathrm { c } } ( \overrightarrow { p } - \overrightarrow { \rho } ) \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \overrightarrow { \rho } ) \mathrm { d } \overrightarrow { \rho }\tag{100}
$$

And the marginal mean of irradiance as (using ergodicity):

$$
\mathbb { E } [ \mathrm { E } ( \vec { p } ) ] = \int _ { \mathbb { R } ^ { 2 } } \mathrm { P } _ { \mathrm { c } } ( \vec { p } - \vec { \rho } ) \mathbb { E } _ { \mathrm { h } } \left[ \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \vec { \rho } ) \right] \mathrm { d } \vec { \rho } .\tag{101}
$$

$$
= \mu _ { \mathcal { L } ^ { ( \Delta _ { \mathrm { i } } ) } } = \mu _ { E }\tag{102}
$$

since the PSF integrates to unity. The conditional variance

$$
\mathbb { V } \Big [ \mathrm { E } \mid \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } \Big ] = \mathbb { V } \left[ \sum _ { m } \bigl ( \varDelta _ { \mathrm { i } } ^ { 2 } \mathrm { P _ { c } } ( \overrightarrow { p } - \overrightarrow { q } _ { m } ) \bigr ) \mathcal { L } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } ) \right]\tag{103}
$$

$$
= \sum _ { m } \left( \varDelta _ { \mathrm { i } } ^ { 2 } \mathrm { P _ { c } } ( \overrightarrow { p } - \overrightarrow { q } _ { m } ) \right) ^ { 2 } \mathbb { V } \left[ \mathcal { L } ^ { ( \varDelta _ { \mathrm { i } } ) } \mid \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } \right]\tag{104}
$$

$$
= \sum _ { m } \bigl ( \varDelta _ { \mathrm { i } } ^ { 2 } \mathrm { P _ { c } } ( \overrightarrow { p } - \overrightarrow { q } _ { m } ) \bigr ) ^ { 2 } \frac { \Bigl ( \overrightarrow { \mathcal { L } } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } ) \Bigr ) ^ { 2 } } { M }\tag{105}
$$

using conditional independence of $\mathcal { L } ^ { ( \Delta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } ) \mid \overline { { \mathcal { L } } } ^ { ( \Delta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } )$ across grid locations. Therefore the marginal variance of irradiance is given by

$$
\mathbb { V } [ \mathrm { E } ] = \mathbb { V } \Big [ \mathbb { E } \Big [ \mathrm { E \ | \ } \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } \Big ] \Big ] + \mathbb { E } \Big [ \mathbb { V } \Big [ \mathrm { E \ | \ } \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } \Big ] \Big ]\tag{106}
$$

The first term

$$
\mathbb { V } \Big [ \mathbb { E } \Big [ \big \mathrm { E ~ } | \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } \Big ] \Big ] = \mathbb { V } \Bigg [ \sum _ { m } \varDelta _ { \mathrm { i } } ^ { 2 } \mathrm { P _ { c } } ( \overrightarrow { p } - \overrightarrow { q } _ { m } ) \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } ) \Bigg ]\tag{107}
$$

$$
= \mathbb { V } [ \overline { { \mathrm { E } } } ( \vec { p } ) ]\tag{108}
$$

Further simplification isn’t possible without information about the correlation structure of first-order texture $\overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \vec { q } _ { m } )$ across grid locations, which are not

necessarily independent. We will therefore leave it as is. The second term

$$
\mathbb { E } \Big [ \mathbb { V } \Big [ \mathrm { E ~ } | \ \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } \Big ] \Big ] = \mathbb { E } \left[ \sum _ { m } \big ( \varDelta _ { \mathrm { i } } ^ { 2 } \mathrm { P _ { c } } ( \overrightarrow { p } - \overrightarrow { q } _ { m } ) \big ) ^ { 2 } \frac { \Big ( \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ( \overrightarrow { q } _ { m } ) \Big ) ^ { 2 } } { M } \right]\tag{109}
$$

$$
= { \frac { 1 } { M } } \sum _ { m } \bigl ( \varDelta _ { \mathrm { i } } ^ { 2 } \mathrm { P } _ { \mathrm { c } } ( { \overrightarrow { p } } - { \overrightarrow { q } } _ { m } ) \bigr ) ^ { 2 } \mathbb { E } \biggl [ \Bigl ( { \overrightarrow { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } ( { \overrightarrow { q } } _ { m } ) \Bigr ) ^ { 2 } \biggr ]\tag{110}
$$

$$
= \mathbb { E } \bigg [ \Big ( \overline { { \mathcal { L } } } ^ { ( \Delta _ { \mathrm { i } } ) } \Big ) ^ { 2 } \bigg ] \frac { \varDelta _ { \mathrm { i } } ^ { 2 } } { M } \sum _ { m } { \varDelta _ { \mathrm { i } } ^ { 2 } } \big ( \mathrm { P } _ { \mathrm { c } } \big ( \overrightarrow { p } - \overrightarrow { q } _ { m } \big ) \big ) ^ { 2 }\tag{111}
$$

$$
\approx \mathbb { E } \left[ \left( \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } \right) ^ { 2 } \right] \frac { \varDelta _ { \mathrm { i } } ^ { 2 } } { M } \int _ { \mathbb { R } ^ { 2 } } ( \mathrm { P _ { c } } ( \overrightarrow { p } - \overrightarrow { \rho } ) ) ^ { 2 } \mathrm { d } \overrightarrow { \rho }\tag{112}
$$

Taking the PSF to be a Gaussian of width $\varDelta _ { \mathrm { c } }$ (with parameter $\varDelta _ { \mathrm { c } } / 2 )$ centered at $\overrightarrow { p }$

$$
\mathrm { P _ { c } } ( \vec { p } - \vec { \rho } ) = \frac { 2 } { \pi \varDelta _ { \mathrm { c } } ^ { 2 } } \exp \left( - 2 \frac { \| \vec { p } - \vec { \rho } \| ^ { 2 } } { \varDelta _ { \mathrm { c } } ^ { 2 } } \right)\tag{113}
$$

$$
\int _ { \mathbb { R } ^ { 2 } } \bigl ( \mathrm { P } _ { \mathrm { c } } \bigl ( \vec { p } - \vec { \rho } \bigr ) \bigr ) ^ { 2 } \mathrm { d } \vec { \rho } = \frac { 1 } { \pi \varDelta _ { \mathrm { c } } ^ { 2 } }\tag{114}
$$

and thus the pointwise marginal variance of irradiance

$$
\mathbb { V } [ \mathrm { E } ( \vec { p } ) ] = \frac { 1 } { M } \left( \frac { \varDelta _ { \mathrm { i } } ^ { 2 } } { \pi \varDelta _ { \mathrm { c } } ^ { 2 } } \right) \mathbb { E } \left[ \left( \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } \right) ^ { 2 } \right] + \mathbb { V } \left[ \overline { { \mathrm { E } } } ( \vec { p } ) \right]\tag{115}
$$

$$
= \frac { 1 } { M } \biggl ( \frac { \varDelta _ { \mathrm { i } } ^ { 2 } } { \pi \varDelta _ { \mathrm { c } } ^ { 2 } } \biggr ) \biggl ( \mu _ { \mathcal { L } ^ { ( \varDelta _ { \mathrm { i } } ) } } ^ { 2 } + \mathbb { V } \bigl [ \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } \bigr ] \biggr ) + \mathbb { V } \bigl [ \overline { { \mathrm { E } } } ( \overrightarrow { p } ) \bigr ]\tag{116}
$$

Therefore the coeficient of variation of irradiance is given by

$$
\kappa _ { \mathrm { E } } ^ { 2 } = \frac { \mathbb { V } [ \mathrm { E } ( \vec { p } ) ] } { \mathbb { E } \left[ \mathrm { E } ( \vec { p } ) \right] ^ { 2 } }\tag{117}
$$

$$
= \frac { 1 } { M } \Bigg ( \frac { \varDelta _ { \mathrm { i } } ^ { 2 } } { \pi \varDelta _ { \mathrm { c } } ^ { 2 } } \Bigg ) \left( 1 + \frac { \mathbb { V } \Big [ \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } \Big ] } { \mathbb { E } \Big [ \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } \Big ] ^ { 2 } } \right) + \frac { \mathbb { V } \big [ \overline { { \mathrm { E } } } \big ] } { \mathbb { E } \big [ \overline { { \mathrm { E } } } \big ] ^ { 2 } }\tag{118}
$$

$$
= \frac { 1 } { M Q } \Big ( 1 + \kappa _ { \overline { { \mathcal { L } } } ^ { ( \Delta _ { \mathrm { i } } ) } } ^ { 2 } \Big ) + \kappa _ { \overline { { \mathrm { E } } } } ^ { 2 }\tag{119}
$$

since $\mu _ { E } = \mathbb { E } \big [ \overline { { \mathbf { E } } } ( \vec { p } ) \big ] = \mathbb { E } \Big [ \overline { { \mathcal { L } } } ^ { ( \varDelta _ { \mathrm { i } } ) } \Big ] = \mu _ { \mathcal { L } ^ { ( \varDelta _ { \mathrm { i } } ) } }$ , and defining $\begin{array} { r } { Q = \frac { \pi \varDelta _ { \mathrm { c } } ^ { 2 } } { \varDelta _ { \mathrm { i } } ^ { 2 } } } \end{array}$ as the efective number of coherence areas contributing to the irradiance at a sensor point through the PSF.

## G Noise model

The (noisy) intensity $\widetilde { \mathrm { I } }$ (in DN, Digital Numbers) recorded by the sensor given an incident flux $\Phi \ [ \mathrm { W } ]$ is modeled as follows

$$
\begin{array} { r l r } {  { \widetilde { \mathrm { I } } ( \vec { p } ) = \mathrm { m i n } ( \lfloor { 1 / g } ( \mathrm { m i n } ( P , P _ { \mathrm { m a x } } ) + Z _ { \mathrm { p r e } } ) + Z _ { \mathrm { p o s t } } \rfloor , \mathrm { I } _ { \mathrm { m a x } } ) } } \\ & { } & { P \downarrow \Phi \sim \mathrm { P o i s s o n } ( ( \zeta \Phi + D ) t ) } \\ & { } & { Z _ { \mathrm { p r e } } \sim { \mathcal N } ( 0 , \sigma _ { \mathrm { p r e } } ^ { 2 } ) , \quad [ { \mathrm { e } } ^ { - } ] } \\ & { } & { Z _ { \mathrm { p o s t } } ( \vec { p } ) \sim { \mathcal N } ( 0 , \sigma _ { \mathrm { p o s t } } ^ { 2 } ) , \quad [ \mathrm { D N } ] } \\ & { } & { \zeta = \displaystyle \frac { v } { K } , \quad [ \mathrm { p h o t o e } ^ { - } \mathrm { J } ^ { - 1 } ] } \\ & { } & { K = \displaystyle \frac { h c _ { 0 } } { \lambda } , \quad [ \mathrm { J } ] } \end{array}\tag{120}
$$

where, $P _ { \mathrm { m a x } }$ is the full-well capacity of the pixel (in electrons), $\operatorname { I } _ { \operatorname* { m a x } }$ is the maximum ADC value (in DN), $1 / g$ is the gain factor (in $[ \mathrm { D N } / \mathrm { e } ^ { - } ] )$ determined by the ISO setting, υ is the quantum eficiency, $D$ is the dark current (in $\left[ \mathrm { e ^ { - } s ^ { - 1 } } \right] ,$ ), t is the exposure time $( \mathrm { i n [ s ] } ) , Z _ { \mathrm { p r e } }$ is the pre-amplifier read noise (in electrons), and $Z _ { \mathrm { p o s t } }$ is the post-amplifier read noise (in DN) separate from noise due to ADC quantization. Parameters of the above noise model used for our numerical studies for Figure 7 are specified in Table $\mathrm { 3 c }$

Simplifying assumptions for analysis. For analytical tractability of (noisy) focus measure statistics, we make the following simplifying assumptions:

1. We operate in a regime where saturation efects (full-well and ADC) can be neglected.

2. We ignore quantization efects, i.e., we consider $\widetilde { \mathrm { I } }$ to be a continuous random variable.

3. We consider dark current to be negligible, i.e., $D \approx 0$

Resulting in the simplified model:

$$
\begin{array} { l } { { \widetilde { \mathrm { I } } ( \overrightarrow { p } ) = \mathbb { 1 } / g ( P + Z _ { \mathrm { p r e } } ) + Z _ { \mathrm { p o s t } } } } \\ { { P \mid \Phi \sim \mathrm { P o i s s o n } \left( \zeta \Phi t \right) } } \end{array}\tag{121}
$$

Under the above assumptions, we define the noise-free intensity I as the conditional mean of I given the incident flux Φ:

$$
\mathrm { I } : = \mathbb { E } \left[ \mathrm { \widetilde { I } } \mid \Phi \right] = { \frac { \zeta t } { g } } \Phi\tag{122}
$$

$$
\mathbb { V } \Big [ \tilde { \ I } \ | \ \Phi \Big ] = \mathbb { V } \Big [ \tilde { \ I } \ | \ \ I \Big ] = \frac { 1 } { g ^ { 2 } } \big [ \zeta t \Phi + \sigma _ { \mathrm { p r e } } ^ { 2 } \big ] + \sigma _ { \mathrm { p o s t } } ^ { 2 }\tag{123}
$$

$$
= \frac { \mathrm { I } } { g } + \frac { \sigma _ { \mathrm { p r e } } ^ { 2 } } { g ^ { 2 } } + \sigma _ { \mathrm { p o s t } } ^ { 2 }\tag{124}
$$

We may rewrite Equation (121) in an additive form, defining $n : = \widetilde { \mathrm { I } } - \mathrm { I }$

$$
{ \begin{array} { c } { { \widetilde { \mathrm { I } } } = \mathrm { I } + n } \\ { \displaystyle \mathbb { V } [ n \mid \mathrm { I } ] = \mathbb { V } { \biggl [ } { \widetilde { \mathrm { I } } } \mid \mathrm { I } { \biggr ] } = { \frac { \mathrm { I } } { g } } + { \frac { \sigma _ { \mathrm { p r e } } ^ { 2 } } { g ^ { 2 } } } + \sigma _ { \mathrm { p o s t } } ^ { 2 } } \end{array} }\tag{125}
$$

which is the model described in $\ S \ O 3$ (excluding saturation).

4. As a final simplification, we assume the incident irradiance $\operatorname { E } ( \vec { p } ) \ [ \mathrm { W } \mathrm { m } ^ { - 2 } ]$ is uniform over the pixel area

$$
\Phi = \mathrm { E } ( \vec { p } ) \varDelta _ { \mathrm { p } } ^ { 2 } , \quad \mathrm { [ W ] }\tag{126}
$$

where $\varDelta _ { \mathrm { p } } ^ { 2 }$ is the pixel area $\mathrm { ( i n ~ [ m ^ { 2 } ] ) }$ . This therefore implies the noise-free intensity is proportional to the incident irradiance as follows:

$$
\mathrm { I } = \frac { \beta t } { g } \mathrm { E } ( \vec { p } )\tag{127}
$$

$$
\beta : = \frac { v \varDelta _ { \mathrm { p } } ^ { 2 } } { K } , \quad \mathrm { [ p h o t o e ^ { - } J ^ { - 1 } m ^ { 2 } ] }\tag{128}
$$

We may also write the conditional mean and variance of $\widetilde { \mathrm { I } }$ given E as:

$$
\begin{array} { r l } & { \mathbb { E } \bigg [ \widetilde { \mathbf { I } } \mid \mathbf { E } \bigg ] = \frac { \beta t } { g } \mathbf { E } } \\ & { \mathbb { V } \bigg [ \widetilde { \mathbf { I } } \mid \mathbf { E } \bigg ] = \frac { 1 } { g ^ { 2 } } \big [ \beta t \mathbf { E } + \sigma _ { \mathrm { p r e } } ^ { 2 } \big ] + \sigma _ { \mathrm { p o s t } } ^ { 2 } } \end{array}\tag{129}
$$

We do not make the above assumptions (other than proportionality of flux to irradiance, Item 4) in our numerical implementation Figure 7, which is implemented according to Equation (120) with practical values chosen for sensor parameters including full-well capacity, dark current etc., that can be found in Table 3c. Further details of the numerical implementation can be found in § D.3.

## H Depth from focus analysis

As described in $\ S \ 3 ,$ , given pixel intensity measurements in a patch $\mathrm { P } _ { T }$ of $T$ pixels, the squared sample coeficient of variation focus measure

$$
\widetilde { \mathrm { c } } _ { j } ^ { 2 } ( \vec { p } ) : = \frac { \widetilde { \mathrm { s } } _ { j } ^ { 2 } ( \vec { p } ) } { \widetilde { \mathrm { m } } _ { j } ( \vec { p } ) } ,\tag{130}
$$

is computed from the sample variance and sample mean of patch intensities,

$$
\widetilde { \mathrm { m } } _ { j } ( \overrightarrow { p } ) : = \frac { 1 } { T } \sum _ { \overrightarrow { p } ^ { \prime } \in \mathrm { P } _ { T } ( \overrightarrow { p } ) } \widetilde { \mathrm { I } } _ { j } ( \overrightarrow { p } ^ { \prime } ) ,\tag{131}
$$

$$
\widetilde { \mathrm { s } } _ { j } ^ { 2 } ( \vec { p } ) : = \frac { 1 } { T - 1 } \sum _ { \vec { p } ^ { \prime } \in \mathrm { P } _ { T } ( \vec { p } ) } \left( \widetilde { \mathrm { I } } _ { j } ( \vec { p } ^ { \prime } ) - \widetilde { \mathrm { m } } _ { j } ( \vec { p } ) \right) ^ { 2 }\tag{132}
$$

The noise-free focus measure $\mathrm { c ^ { 2 } }$ is defined similarly :

$$
\mathrm { c } _ { j } ^ { 2 } ( \vec { p } ) : = \frac { \mathrm { s } _ { j } ^ { 2 } ( \vec { p } ) } { \mathrm { m } _ { j } ( \vec { p } ) } ,\tag{133}
$$

$$
\mathrm { s } _ { j } ^ { 2 } ( \vec { p } ) : = \frac { 1 } { T - 1 } { \sum _ { \vec { p } ^ { \prime } \in \mathrm { P } _ { T } ( \vec { p } ) } } \big ( \mathrm { I } _ { j } \big ( \vec { p } ^ { \prime } \big ) - \mathrm { m } _ { j } \big ( \vec { p } \big ) \big ) ^ { 2 }\tag{134}
$$

$$
\mathrm { m } _ { j } ( \vec { p } ) : = \frac { 1 } { T } \sum _ { \vec { p } ^ { \prime } \in \mathrm { P } _ { T } ( \vec { p } ) } \mathrm { I } _ { j } ( \vec { p } ^ { \prime } ) ,\tag{135}
$$

For the purposes of analysis, we also define a normalized version of the sample variance

$$
\widetilde \mathrm { C } ^ { 2 } ( \overrightarrow { p } ) : = \frac { \widetilde { \mathrm { s } } ^ { 2 } ( \overrightarrow { p } ) } { \mu _ { \mathrm { I } } ^ { 2 } }\tag{136}
$$

$$
\mathrm { C } ^ { 2 } ( \vec { p } ) : = \frac { \mathrm { s } ^ { 2 } ( \vec { p } ) } { \mu _ { \mathrm { I } } ^ { 2 } }\tag{137}
$$

where $\mu _ { \mathrm { I } } = \mathbb { E } [ \mathrm { I } ] = \mathbb { E } \bigg [ \bigg ]$ is the marginal mean of intensities over the patch. This enormalized sample variance is distinct from the actual focus measure defined above (and not directly measureable exactly). For analytical tractability, we study the statistics of $\widetilde { \mathrm { C } } ^ { 2 }$ instead of ${ \widetilde { \mathrm { c } } } ^ { 2 }$ , to approximately characterize ${ \widetilde { \mathrm { c } } } ^ { 2 }$

## eH.1 Statistics of the focus measure

Given the simplified sensor noise model in Equation (125), we can denote the pointwise marginal mean and variance of measured (noisy) pixel intensity I as

$$
\mu _ { \widetilde { \mathrm { I } } } : = \mathbb { E } \left[ \widetilde { \mathrm { I } } \right] = \mu _ { \mathrm { I } }\tag{138}
$$

$$
\sigma _ { \widetilde { \mathrm { I } } } ^ { 2 } : = \mathbb { V } \Big [ \widetilde { \mathrm { I } } \Big ] = \sigma _ { \mathrm { I } } ^ { 2 } + \frac { \mu _ { \mathrm { I } } } { g } + \frac { \sigma _ { \mathrm { p r e } } ^ { 2 } } { g ^ { 2 } } + \sigma _ { \mathrm { p o s t } } ^ { 2 }\tag{139}
$$

where the marginal mean and variance of true intensities (as a consequence of assuming uniform irradiance over the pixel area, Equation (127)) are given by

$$
\mu _ { \mathrm { I } } : = \mathbb { E } [ \mathrm { I } ] = \frac { \beta t } { g } \mu _ { E }\tag{140}
$$

$$
\sigma _ { \mathrm { I } } ^ { 2 } : = \mathbb { V } [ \mathrm { I } ] = \frac { \beta ^ { 2 } t ^ { 2 } } { g ^ { 2 } } \sigma _ { E } ^ { 2 }\tag{141}
$$

and as a consequence of this assumption,

$$
\kappa _ { \mathrm { I } } ^ { 2 } : = { \frac { \sigma _ { \mathrm { I } } ^ { 2 } } { \mu _ { \mathrm { I } } ^ { 2 } } } = { \frac { \sigma _ { E } ^ { 2 } } { \mu _ { E } ^ { 2 } } } = \kappa _ { \mathrm { E } } ^ { 2 }\tag{142}
$$

We can now write the coeficient of variation of noisy intensity as

$$
\kappa _ { \widetilde { \mathrm { I } } } ^ { 2 } : = \frac { \sigma _ { \widetilde { \mathrm { I } } } ^ { 2 } } { \mu _ { \widetilde { \mathrm { I } } } ^ { 2 } } = \kappa _ { \mathrm { I } } ^ { 2 } + \kappa _ { n } ^ { 2 }\tag{143}
$$

$$
= \kappa _ { \mathrm { E } } ^ { 2 } + \kappa _ { n } ^ { 2 }\tag{144}
$$

where we defined in Equation (2) the squared reciprocal of pixel intensity SNR as the coeficient of variation of noise $\kappa _ { n } ^ { 2 }$ :

$$
\kappa _ { n } ^ { 2 } : = \frac { \sigma _ { n } ^ { 2 } } { \mu _ { \mathrm { I } } ^ { 2 } } = \frac { 1 } { t \zeta \mu _ { \Phi } } \biggl ( 1 + \frac { \sigma _ { \mathrm { r e a d } } ^ { 2 } } { t \zeta \mu _ { \Phi } } \biggr )\tag{rep. 2}
$$

with the marginal variance of noise $n _ { : }$

$$
\sigma _ { n } ^ { 2 } : = \mathbb { V } [ n ] = \mathbb { E } [ \mathbb { V } [ n \mid \mathrm { I } ] ] + \mathbb { V } [ 0 ] = { \frac { \mu _ { \mathrm { I } } } { g } } + { \frac { \sigma _ { \mathrm { p r e } } ^ { 2 } } { g ^ { 2 } } } + \sigma _ { \mathrm { p o s t } } ^ { 2 }\tag{145}
$$

using the conditional variance from Equation (125). The pointwise marginal distribution of pixel intensity $\widetilde { \mathrm { I } }$ at each pixel is identical across pixels. We make ethe additional assumption that these pixel intensities are independent across pixels.<sup>12</sup> Therefore the sample variance of noisy pixel intensities has marginal statistics:

$$
\begin{array} { l } { { \mathbb { E } \left[ \widetilde { \mathbf { s } } ^ { 2 } \right] = \mathbb { V } \left[ \widetilde { \mathbf { I } } \right] = \sigma _ { \widetilde { \mathbf { I } } } ^ { 2 } } } \\ { { \mathbb { V } \left[ \widetilde { \mathbf { s } } ^ { 2 } \right] = \displaystyle \frac { \left( \sigma _ { \widetilde { \mathbf { I } } } ^ { 2 } \right) ^ { 2 } } { T } \left( \kappa _ { 4 } - \frac { T - 3 } { T - 1 } \right) } } \end{array}
$$

where $\kappa _ { 4 }$ is the kurtosis (ratio of fourth moment to squared-variance) of the marginal distribution of pixel intensities. If the distribution of pixel intensities is approximated to be to Gaussian, then $\kappa _ { 4 } \approx 3$

Normalizing by $\mu _ { \widetilde { \mathrm { I } } } ^ { 2 }$ we then have the marginal statistics of $\widetilde { \mathrm { C } } ^ { 2 }$

Lemma 11: Marginal statistics of $\widetilde { \mathrm { C } } ^ { 2 }$

$$
\mathbb { E } \Big [ \widetilde { \mathbf { C } } ^ { 2 } \Big ] = \frac { \sigma _ { \widetilde { \mathrm { I } } } ^ { 2 } } { \mu _ { \widetilde { \mathrm { I } } } ^ { 2 } } = \kappa _ { \widetilde { \mathrm { I } } } ^ { 2 } = \kappa _ { \mathrm { I } } ^ { 2 } + \kappa _ { n } ^ { 2 }\tag{146}
$$

$$
\mathbb { V } \Big [ \widetilde { \mathbf { C } } ^ { 2 } \Big ] = \frac { \Big ( \kappa _ { \widetilde { \mathrm { I } } } ^ { 2 } \Big ) ^ { 2 } } { T } \Big ( \kappa _ { 4 } - \frac { T - 3 } { T - 1 } \Big )\tag{147}
$$

$$
= { \frac { 2 { \Big ( } \kappa _ { \widetilde { \mathrm { I } } } ^ { 2 } { \Big ) } ^ { 2 } } { T - 1 } } \quad ( { \mathrm { i f ~ } } \kappa _ { 4 } \approx 3 , { \mathrm { ~ e f f e c t i v e l y ~ } } \widetilde { \mathrm { I } } \sim { \mathcal { N } } { \big ( } \mu _ { \mathrm { I } } , \sigma _ { n } ^ { 2 } { \big ) } )\tag{148}
$$

Statistics of focus measure for first-order textureless surface. For a first-order textureless surface, we have $\kappa _ { \mathrm { I } } ^ { 2 } = \dot { \kappa } _ { \mathrm { E } } ^ { 2 } = 1 / Q M$ , and so we have

$$
\mathbb { E } \left[ \widetilde { \mathrm { C } } ^ { 2 } \right] = \frac { 1 } { Q M } + \kappa _ { n } ^ { 2 }
$$

$$
\mathbb { V } \bigg [ \widetilde { \mathbf { C } } ^ { 2 } \bigg ] = \frac { 2 } { T - 1 } \Bigg \{ \frac { 1 } { Q M } + \kappa _ { n } ^ { 2 } \Bigg \} ^ { 2 }
$$

## H.2 Proof of Proposition 2

§ H.1 establishes the mean and variance of $\widetilde { \mathrm { C } } ^ { 2 }$ . We now present a proof for eProposition 2, characterizing the α-recoverability of a surface from depth from focus, as defined in Definition 1.

We indicate the in-focus and out-of-focus normalized sample variances as $\mathrm { \widetilde { C } _ { i f } ^ { 2 } }$ and $\widetilde \mathrm { C } _ { \mathrm { o o f } } ^ { 2 }$ respectively, and the corresponding noise-free focus measures as $\mathrm { C _ { i f } ^ { 2 } }$ eand $\mathrm { C _ { o o f } ^ { 2 } } .$

1. $\mathrm { \widetilde { C } _ { i f } ^ { 2 } }$ is the value of $\widetilde { \mathrm { C } } ^ { 2 }$ evaluated at the sharpest focus setting of the camera.

2. $\widetilde \mathrm { C } _ { \mathrm { o o f } } ^ { 2 }$ eis the value of $\widetilde { \mathrm { C } } ^ { 2 }$ evaluated at a focus setting of the camera that e eis suficiently far from the in-focus setting, such that the true noise-free intensities in the patch are efectively constant $( i . e .$ , the texture is completely blurred out), or $\operatorname { I } ( { \vec { p } } ) \approx \mu _ { \mathrm { I } }$ . This therefore implies $\mathrm { C _ { o o f } ^ { 2 } = 0 }$

To proceed, we model $\mathrm { \widetilde { C } _ { i f } ^ { 2 } }$ and $\widetilde \mathrm { C } _ { \mathrm { o o f } } ^ { 2 }$ in and out of focus to be normallye edistributed random variables, with the mean and variance expressions derived in $\ S$ H.1. The conclusions of this analysis are supported by our Monte-Carlo simulations in Figure $^ { 7 , }$ thereby justifying the use of this assumption for the purpose of analytical tractability.

Modeling $\mathrm { \widetilde { C } _ { i f } ^ { 2 } }$ and $\widetilde \mathrm { C } _ { \mathrm { o o f } } ^ { 2 }$ to be normally-distributed, with the mean and variance ederived in Lemma 11.

$$
\widetilde { \mathrm { C } } _ { \mathrm { i f } } ^ { 2 } \sim { \mathcal N } \bigg ( \kappa _ { \mathrm { I } } ^ { 2 } + \kappa _ { n } ^ { 2 } , \quad \frac { 2 } { T - 1 } \big ( \kappa _ { \mathrm { I } } ^ { 2 } + \kappa _ { n } ^ { 2 } \big ) ^ { 2 } \bigg )\tag{149}
$$

$$
\widetilde { \mathrm { C } } _ { \mathrm { o o f } } ^ { 2 } \sim \mathcal { N } \bigg ( \kappa _ { n } ^ { 2 } , \quad \frac { 2 } { T - 1 } \big ( \kappa _ { n } ^ { 2 } \big ) ^ { 2 } \bigg )\tag{150}
$$

In addition, since the out-of-focus intensity variations in $\widetilde \mathrm { C } _ { \mathrm { o o f } } ^ { 2 }$ are purely due to sensor noise, we can say that $\mathrm { \widetilde { C } _ { i f } ^ { 2 } }$ and $\widetilde \mathrm { C } _ { \mathrm { o o f } } ^ { 2 }$ eare uncorrelated random variables. Therefore,

$$
\begin{array} { r l } & { \mathbb { P } \mathbf { r } \left( \widetilde { C } _ { \mathbf { q } } ^ { 2 } \succ \widetilde { C } _ { \mathrm { o s f } } ^ { 2 } \right) = \phi \left( \frac { \mathbb { E } \left[ \widetilde { C } _ { \mathbf { q } } ^ { 2 } \right] - \mathbb { E } \left[ \widetilde { C } _ { \mathrm { o s f } } ^ { 2 } \right] } { \sqrt { \Psi \left[ \widetilde { C } _ { \mathbf { q } } ^ { 2 } \right] + \Psi \left[ \widetilde { C } _ { \mathrm { o s f } } ^ { 2 } \right] } } \right) } \\ & { \quad \quad = \bar { \Phi } \left( \frac { \kappa _ { 1 } ^ { 2 } } { \sqrt { \tau _ { - 1 } ^ { 2 } \left\{ \left( \kappa _ { 1 } ^ { 2 } + \kappa _ { n } ^ { 2 } \right) ^ { 2 } + \left( \kappa _ { n } ^ { 2 } \right) ^ { 2 } \right\} } } \right) } \\ & { \quad \quad \quad = \phi \left( \frac { \theta ^ { 2 } } { \sqrt { \tau _ { - 1 } ^ { 2 } \left\{ \left( \Theta ^ { 2 } + 1 \right) ^ { 2 } + 1 \right\} } } \right) } \end{array}\tag{151}
$$

(152)

(153)

where $\varPhi$ is the CDF of the standard normal distribution, and we’ve defined $\Theta ^ { 2 }$ in $\ S \ 5 .$

$$
\Theta ^ { 2 } : = \frac { \kappa _ { \mathrm { I } } ^ { 2 } } { \kappa _ { n } ^ { 2 } }\tag{154}
$$

## Lemma 12: Probability of Error

Modeling $\mathrm { \widetilde { C } _ { i f } ^ { 2 } }$ and $\widetilde \mathrm { C } _ { \mathrm { o o f } } ^ { 2 }$ to be normally-distributed, with the mean and variance e ederived in Lemma 11, the probability of error

$$
\mathrm { P r } \Big ( \widetilde { \mathrm { C } } _ { \mathrm { i f } } ^ { 2 } < \widetilde { \mathrm { C } } _ { \mathrm { o o f } } ^ { 2 } \Big ) = 1 - \mathscr { F } \left( \frac { \theta ^ { 2 } } { \sqrt { \frac { 2 } { T - 1 } \Big \{ \left( \theta ^ { 2 } + 1 \right) ^ { 2 } + 1 \Big \} } } \right)\tag{155}
$$

Therefore we have a necessary and suficient condition for α-recoverability (Definition 1) of a surface using depth from focus:

$$
\frac { \theta ^ { 2 } } { \sqrt { 1 + \left( \theta ^ { 2 } + 1 \right) ^ { 2 } } } \cdot \sqrt { \frac { T - 1 } { 2 } } > \phi ^ { - 1 } ( 1 - \alpha )\tag{156}
$$

thus proving Proposition 2.

Notice that as Θ is increased, the LHS saturates to $\scriptstyle { \sqrt { \frac { T - 1 } { 2 } } }$ , which is the maximum possible value of the argument of the CDF. Therefore, for a fixed value of $T _ { : }$ , there is a maximum achievable probability of correct recovery, which is given by $\Phi { \left( \sqrt { \frac { T - 1 } { 2 } } \right) }$ .

## I Recoverability with second-order texture

In this section, we derive Proposition 7, which characterizes the contrast-SNR product Θ as a function of exposure time t and bandwidth $\varDelta _ { k }$ , for a first-order textureless surface. For a first-order textureless surface, we have $\kappa _ { \overline { { \mathrm { E } } } } ^ { 2 } = 0$ , and so from Proposition 6

$$
\kappa _ { \mathrm { I } } ^ { 2 } = { \frac { 1 } { Q M } }\tag{157}
$$

with $M \approx G _ { 1 } \varDelta _ { k }$ (Equation (81)). We can similarly expand $\kappa _ { n } ^ { 2 }$ using the noise model in Equation (2).

$$
\kappa _ { n } ^ { 2 } = \frac { 1 } { t \zeta \mu _ { \Phi } } \biggl ( 1 + \frac { \sigma _ { \mathrm { r e a d } } ^ { 2 } } { t \zeta \mu _ { \Phi } } \biggr )\tag{158}
$$

Spectral filtering reduces the mean flux $\mu _ { \Phi }$ by a factor of $\gamma _ { \mathrm { g } }$ (described in § I.1), which is a function of the filter bandwidth $\varDelta _ { k }$

$$
\gamma _ { \mathrm { g } } : = \frac { \mu _ { \Phi } ^ { 1 } } { \mu _ { \Phi } ^ { 0 } } \approx \mathrm { g } _ { \mathrm { p } } \frac { \varDelta _ { k } } { \varDelta _ { \mathrm { s r c } } }\tag{159}
$$

where $\mu _ { \Phi } ^ { 0 }$ and $\mu _ { \Phi } ^ { 1 }$ are the mean flux before and after spectral filtering respectively, and $\mathrm { g _ { p } }$ is the peak attenuation of the spectral filter. For suficiently narrow spectral filter bandwidth, the efective bandwidth after filtering is approximately the same as that of the filter, and so we abuse notation and use $\varDelta _ { k }$ for both the spectral filter bandwidth as well as the efective bandwidth after filtering. We denote the fixed bandwidth of the original (unfiltered) source spectrum by $\varDelta _ { \mathrm { s r c } }$ (written $\varDelta _ { k }$ in Equation (80)). These are clarified in § I.1.

We define

$$
G _ { 2 } : = \frac { \zeta \mu _ { \Phi } ^ { 0 } \cdot \mathrm { g } _ { \mathrm { p } } } { \varDelta _ { \mathrm { s r c } } }\tag{160}
$$

and plugging in $\zeta \mu _ { \Phi } ^ { 1 } = G _ { 2 } \varDelta _ { k }$ into the expression for $\kappa _ { n } ^ { 2 }$ , we have the contrast-SNR product for a first-order textureless surface as a function of exposure time and (filter) bandwidth

$$
\theta ^ { 2 } ( t , \Delta k ) \approx \frac { 1 } { Q } \cdot \frac { G _ { 2 } } { G _ { 1 } } \cdot \frac { t } { \left( 1 + \frac { \sigma _ { \mathrm { r e a d } } ^ { 2 } } { G _ { 2 } } \frac { 1 } { \Delta _ { k } t } \right) }\tag{161}
$$

Since bandwidth afects $\mu _ { \Phi } ^ { 1 }$ , the upper limit of exposure time until saturation is bandwidth-dependent. A simple notion to model this is to limit the exposure time until $\begin{array} { r } { \mu _ { \mathrm { I } } = \zeta t { \frac { \mu _ { \Phi } ^ { \mathrm { 1 } } } { g } } = { \frac { G _ { 2 } \varDelta _ { k } } { g } } t } \end{array}$ equals I<sub>max</sub>.

$$
t \in \left[ 0 , \frac { 1 } { \varDelta k } \cdot \frac { g \mathrm { I } _ { \mathrm { m a x } } } { G _ { 2 } } \right)\tag{162}
$$

thus resulting in Proposition 7.

## I.1 Attenuation due to spectral filtering

Consider the original normalized spectral profile (before spectral filtering) is denoted as $\mathrm { f } _ { 0 } ( k )$ , as in Equation (79). Let the attenuation profile of the spectral filter introduced be $\mathrm { g } ( k )$ . Thus the efective attenuation in irradiance due to the spectral filter is given by the factor

$$
\gamma _ { \mathrm { g } } : = \frac { \mu _ { \Phi } ^ { 1 } } { \mu _ { \Phi } ^ { 0 } } = \int _ { \mathbb { R } ^ { + } } \mathrm { g } ( k ) \mathrm { f } _ { 0 } ( k ) \mathrm { d } k\tag{163}
$$

## I.2 Gaussian spectral profile

The spectral profile of various light sources and object reflectance are wide and varied. However, to build intuition for the efect of spectral mismatch between the spectral filter and the illumination and reflectance spectrum, we consider a Gaussian profile for both $\mathrm { f } _ { 0 } ( k )$ and $\mathrm { g } ( k )$

We have previously introduced a Gaussian form for $\mathrm { f _ { 0 } }$ in Equation (80)

$$
\mathrm { f } _ { \mathrm { 0 } } ( \boldsymbol { k } ) = \frac { 1 } { \sqrt { 2 \pi } \varDelta _ { \mathrm { s r c } } } \exp { \left( - \frac { \left( \boldsymbol { k } - \overline { { \boldsymbol { k } } } _ { \mathrm { f _ { 0 } } } \right) ^ { 2 } } { 2 \varDelta _ { \mathrm { s r c } } ^ { 2 } } \right) }\tag{164}
$$

The spectral filter has a peak attenuation $\mathrm { g _ { p } }$

$$
\mathrm { g } ( k ) = \mathrm { g } _ { \mathrm { p } } \exp \left( - \frac { \left( k - \overline { { k } } _ { \mathrm { g } } \right) ^ { 2 } } { 2 \varDelta _ { \mathrm { g } } ^ { 2 } } \right)\tag{165}
$$

The efective spectral profile after introducing the spectral filter will also maintain a Gaussian form,

$$
\mathrm { f } _ { 0 } ( \boldsymbol { k } ) \mathrm { g } ( \boldsymbol { k } ) = \gamma _ { \mathrm { g } } \cdot \left[ \frac { 1 } { \sqrt { 2 \pi } \varDelta _ { \mathrm { e f f } } } \exp \left( - \frac { \left( \boldsymbol { k } - \boldsymbol { \overline { { k } } } _ { \mathrm { e f f } } \right) ^ { 2 } } { 2 \varDelta _ { \mathrm { e f f } } ^ { 2 } } \right) \right]\tag{166}
$$

with efective mean wavenumber and bandwidth given by

$$
\overline { { k } } _ { \mathrm { e f f } } = \overline { { k } } _ { \mathrm { f _ { 0 } } } \frac { \varDelta _ { \mathrm { g } } ^ { 2 } } { \varDelta _ { \mathrm { g } } ^ { 2 } + \varDelta _ { \mathrm { s r c } } ^ { 2 } } + \overline { { k } } _ { \mathrm { g } } \frac { \varDelta _ { \mathrm { s r c } } ^ { 2 } } { \varDelta _ { \mathrm { g } } ^ { 2 } + \varDelta _ { \mathrm { s r c } } ^ { 2 } }\tag{167}
$$

$$
\frac { 1 } { \varDelta _ { \mathrm { e f f } } ^ { 2 } } = \frac { 1 } { \varDelta _ { \mathrm { s r c } } ^ { 2 } } + \frac { 1 } { \varDelta _ { \mathrm { g } } ^ { 2 } }\tag{168}
$$

and efective attenuation factor

$$
\gamma _ { \mathrm { g } } = \mathrm { g _ { p } } \frac { \varDelta _ { \mathrm { g } } } { \sqrt { \varDelta _ { \mathrm { s r c } } ^ { 2 } + \varDelta _ { \mathrm { g } } ^ { 2 } } } \exp { \left( - \frac { \left( \overline { { k } } _ { \mathrm { f _ { 0 } } } - \overline { { k } } _ { \mathrm { g } } \right) ^ { 2 } } { 2 \left( \varDelta _ { \mathrm { s r c } } ^ { 2 } + \varDelta _ { \mathrm { g } } ^ { 2 } \right) } \right) } .\tag{169}
$$

implying a quadratic-exponential attenuation as the filter center frequency $\overline { { k } } _ { \mathrm { g } }$ moves away from the original center frequency $\overline { { k } } _ { \mathrm { f _ { 0 } } }$

Narrowband limit. If we consider a spectral filter with a bandwidth $\varDelta _ { \mathrm { g } } \ll \varDelta _ { \mathrm { s r c } }$ then we can approximate the efective attenuation factor as

$$
\gamma _ { \mathrm { g } } \approx \mathrm { g } _ { \mathrm { p } } \frac { \varDelta _ { \mathrm { g } } } { \varDelta _ { \mathrm { s r c } } } \exp \left( - \frac { \left( \overline { { k } } _ { \mathrm { f _ { 0 } } } - \overline { { k } } _ { \mathrm { g } } \right) ^ { 2 } } { 2 \varDelta _ { \mathrm { s r c } } ^ { 2 } } \right)\tag{170}
$$

which is linear in the filter bandwidth $\varDelta _ { \mathrm { g } } .$ . Further, the efective bandwidth after filtering $\varDelta _ { \mathrm { e f f } }$ is approximately the same as the filter bandwidth $\varDelta _ { \mathrm { g } }$ in this case. If we additionally consider that the width of the original spectrum $\varDelta _ { \mathrm { s r c } }$ is suficiently wide compared to the spectral mismatch $\left( \overline { { k } } _ { \mathrm { f _ { 0 } } } \mathrm { ~ - ~ } \overline { { k } } _ { \mathrm { g } } \right)$ , then we can further approximate the attenuation factor as

$$
\gamma _ { \mathrm { g } } \approx \mathrm { g } _ { \mathrm { p } } \frac { \varDelta _ { \mathrm { g } } } { \varDelta _ { \mathrm { s r c } } } \approx \mathrm { g } _ { \mathrm { p } } \frac { \varDelta _ { \mathrm { e f f } } } { \varDelta _ { \mathrm { s r c } } }\tag{171}
$$

## I.3 Wavenumber and wavelength bandwidths

In our results, we specify bandwidths in terms of wavelength bandwidths as $\varDelta _ { \lambda }$ (specified in nm) because this is the common convention for specification of spectral filters as is more familiar to readers. However, we need to convert this to wavenumber bandwidth $\varDelta _ { k }$ for our analysis and monte-carlo numerical study (Equation (81)). Considering a central wavelength ${ \overline { { \lambda } } } ,$ we consider the mean (angular) wavenumber $\overline { { k } } = { 2 \pi } / { \overline { { \lambda } } }$

$$
k _ { \mathrm { l o w } } = \frac { 2 \pi } { \overline { { \lambda } } + \frac { \varDelta _ { \lambda } } { 2 } }\tag{172}
$$

$$
k _ { \mathrm { h i g h } } = \frac { 2 \pi } { \overline { { \lambda } } - \frac { \varDelta _ { \lambda } } { 2 } }\tag{173}
$$

$$
\varDelta _ { k } = k _ { \mathrm { h i g h } } - k _ { \mathrm { l o w } }\tag{174}
$$

$$
= 2 \pi \cdot \frac { \varDelta _ { \lambda } } { \overline { { { \lambda } } } ^ { 2 } - \frac { \varDelta _ { \lambda } ^ { 2 } } { 4 } }\tag{175}
$$

Under the condition that $\varDelta _ { \lambda } \ll \overline { { \lambda } }$

$$
\varDelta _ { k } \approx \frac { 2 \pi } { \overline { { \lambda } } } \cdot \frac { \varDelta _ { \lambda } } { \overline { { \lambda } } } = \overline { { k } } \cdot \frac { \varDelta _ { \lambda } } { \overline { { \lambda } } }\tag{176}
$$