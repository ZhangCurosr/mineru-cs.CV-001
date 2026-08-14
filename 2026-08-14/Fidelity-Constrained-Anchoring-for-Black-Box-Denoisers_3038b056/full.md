# Fidelity-Constrained Anchoring for Black-Box Denoisers

Masaki Satoh

Morpho, Inc.

Tokyo, Japan

m-satoh@morphoinc.com

Abstract—We propose a fidelity-constrained framework that anchors the output of a black-box denoiser to its input without retraining and with little additional computation. The method linearly blends the denoised image with the input and selects the maximum blending factor that satisfies a prescribed local fidelity constraint using Peak Signal-to-Noise Ratio (PSNR) or Structural Similarity Index (SSIM). For PSNR control, a closed-form solution is obtained under a local constant-blending assumption. For SSIM control, we derive a tractable formulation based on inverse SSIM under the same assumption and solve it efficiently using iterative root finding. Experiments on DIV2K images with synthetic Gaussian noise and outputs from Real-ESRGAN and a non-local means denoiser show that the proposed anchoring strategy provides effective fidelity control while balancing denoising performance and statistical naturalness, as measured by the excess kurtosis of residual noise. In particular, SSIM-based anchoring yields more consistent behavior across noise levels than PSNR-based anchoring.

## I. INTRODUCTION

Denoising is a fundamental task in image processing [1]. Its objective is to improve perceptual quality by removing unwanted noise while preserving structural details. Although deep learning-based denoisers have recently achieved state-ofthe-art performance, they often introduce unnatural artifacts. High-fidelity restoration is essential in applications such as surveillance. In this context, fidelity is defined as the similarity between a denoised output and its input image. Representative fidelity metrics include Peak Signal-to-Noise Ratio (PSNR) and the Structural Similarity Index (SSIM) [2]. These metrics can be incorporated into training losses to improve fidelity. However, such training-time constraints do not guarantee that every output will satisfy a prescribed fidelity condition. Incorporating a denoiser into a feedback loop to enforce fidelity constraints is also possible, but this approach is impractical because directly controlling the output of a complex denoiser is difficult and may require repeated denoiser evaluations.

In this paper, we propose a framework that anchors the output of a black-box denoiser to the input image using fidelity metrics. Avoiding direct manipulation of the denoiser, we linearly blend the denoiser output B with the input image I to produce the output O:

$$
O = I + \alpha \odot ( B - I ) ,\tag{1}
$$

where α is a blending matrix and ⊙ denotes element-wise multiplication. As α increases, O approaches B and fidelity

to the input decreases. The blending factor α is computed to satisfy a fidelity condition between I and $O \mathrm { : }$

$$
\hat { \alpha } = \operatorname* { m a x } \{ \alpha \mid f ( I , O ) \geq T \} ,\tag{2}
$$

where $f ( I , O )$ is the fidelity metric and $T$ is a prescribed threshold. In other words, we seek an operating point that balances fidelity and denoising performance. Because the framework is based on blending, image-quality tuning can be performed without retraining the denoiser, and the proposed method is applicable to a broad range of denoisers.

As fidelity metrics, we consider PSNR and SSIM. Because the Mean Squared Error (MSE) constraint corresponding to PSNR is quadratic in α, we can easily obtain a closed-form solution. In contrast, SSIM is highly nonlinear in α, but we show that its inverse has a tractable form and that the operating point can be found efficiently with a small number of iterations.

## II. FIDELITY METRICS

In this section, we consider two widely used metrics, PSNR and SSIM, and examine how to control them by adjusting the blending factor α.

## A. PSNR

The PSNR between two images I and O is defined as follows:

$$
\mathrm { P S N R } ( I , O ) \equiv 1 0 \log _ { 1 0 } \frac { L ^ { 2 } } { \mathrm { M S E } ( I , O ) } ,\tag{3}
$$

where L is the maximum possible pixel value, and $\mathrm { M S E } ( I , O )$ is the mean squared error between I and O. For 8-bit images, $L \ = \ 2 5 5$ . Therefore, PSNR is monotonically related to MSE and can be controlled through MSE. For simplicity, we consider MSE instead of PSNR, and the problem can be formulated as

$$
\hat { \alpha } = \operatorname* { m a x } \{ \alpha | \mathrm { M S E } ( I , O ) \leq T _ { \mathrm { M S E } } \} ,\tag{4}
$$

where $T _ { \mathrm { M S E } }$ is a threshold on MSE.

MSE is usually computed over the entire image. However, this is too coarse for our purpose, so we compute MSE locally. Specifically, we compute MSE over local subimages of I and O centered at each pixel and impose the constraint on each subimage. We then assign the blending factor computed for each subimage to its corresponding center pixel. Let $i ^ { k } , \ o ^ { k } .$

and $b ^ { k }$ be the k-th subimages of $I , O ;$ , and $B ,$ respectively. We assume that the blending factor α takes a constant value $\alpha ^ { k }$ within each subimage. This working assumption may cause the fidelity condition to be violated in some cases, and we evaluate its validity in the experimental results. Then, the MSE between subimages $i ^ { k }$ and $o ^ { k }$ is

$$
\mathrm { M S E } ^ { k } ( \alpha ^ { k } ) = ( \alpha ^ { k } ) ^ { 2 } \mathrm { M S E } ( i ^ { k } , b ^ { k } ) ,\tag{5}
$$

and the desired value of $\alpha ^ { k }$ is

$$
\hat { \alpha } ^ { k } = \operatorname* { m i n } \left\{ 1 , \sqrt { \frac { T _ { \mathrm { M S E } } } { \mathrm { M S E } ( i ^ { k } , b ^ { k } ) } } \right\} .\tag{6}
$$

Thus, MSE, and therefore PSNR, can be controlled by alpha blending under the constant-α assumption.

The MSE between $i ^ { k }$ and $b ^ { k }$ is expected to be sensitive to the noise level. When the input noise is sufficiently large, the MSE is expected to be large, making the blending factor α small, so the output O almost coincides with I. In contrast, when the input noise is sufficiently small, O almost coincides with B. Thus, we should tune the anchoring condition according to the input noise level. However, this is impossible because the input noise level is unknown in practice. In the following subsection, we show that SSIM is expected to be more robust to noise-level variations.

## B. SSIM

In contrast to PSNR, SSIM is more difficult to control because of its nonlinear form. The SSIM between I and O is defined as follows:

$$
\mathrm { S S I M } ( I , O ) \equiv \frac { ( 2 \mu _ { I } \mu _ { O } + C _ { 1 } ) ( 2 \sigma _ { I O } + C _ { 2 } ) } { ( \mu _ { I } ^ { 2 } + \mu _ { O } ^ { 2 } + C _ { 1 } ) ( \sigma _ { I } ^ { 2 } + \sigma _ { O } ^ { 2 } + C _ { 2 } ) } ,\tag{7}
$$

where $\mu _ { I }$ and $\mu _ { O }$ are the mean intensities of I and $O , \sigma _ { I } ^ { 2 }$ and $\sigma _ { O } ^ { 2 }$ are the variances, $\sigma _ { I O }$ is the covariance, and $C _ { 1 }$ and $C _ { 2 }$ are stabilizing constants. Typical values for 8-bit grayscale images are $C _ { 1 } = ( 0 . 0 1 \times 2 5 5 ) ^ { 2 }$ and $C _ { 2 } = ( 0 . 0 3 \times 2 5 5 ) ^ { 2 }$ . When I and O are identical, $\mathrm { S S I M } ( I , O ) = 1$ ; as their difference increases, $\mathrm { S S I M } ( I , O )$ decreases toward 0. If two images are inversely correlated, SSIM can take negative values.

In practice, the means and (co)variances are computed with small Gaussian windows to obtain local SSIM maps, which are then averaged to produce a single global SSIM value. We impose the fidelity condition on each local SSIM value. For simplicity, we approximate Gaussian-window convolution by considering subimages centered at each pixel, as in the PSNR case. Assuming that the blending factor α takes a constant value $\alpha ^ { k }$ within each subimage, we define the inverse local SSIM as follows:

$$
\psi ^ { k } ( \alpha ) \equiv \frac { 1 } { \mathrm { S S I M } ^ { k } ( \alpha ) } \equiv \psi _ { \mu } ^ { k } ( \alpha ) \psi _ { \sigma } ^ { k } ( \alpha ) ,\tag{8}
$$

where $\psi _ { \mu } ^ { k } ( \alpha )$ and $\psi _ { \sigma } ^ { k } ( \alpha )$ are the $\mu -$ and σ-dependent compo nents of inverse SSIM, respectively, and are given by

$$
\psi _ { \mu } ^ { k } ( \alpha ) = 1 + \frac { \alpha ^ { 2 } ( \mu _ { b } - \mu _ { i } ) ^ { 2 } } { 2 \mu _ { i } ^ { 2 } + 2 \alpha \mu _ { i } ( \mu _ { b } - \mu _ { i } ) + C _ { 1 } }\tag{9}
$$

$$
\psi _ { \sigma } ^ { k } ( \alpha ) = 1 + \frac { \alpha ^ { 2 } ( \sigma _ { i } ^ { 2 } + \sigma _ { b } ^ { 2 } - 2 \sigma _ { i b } ) } { 2 \sigma _ { i } ^ { 2 } + 2 \alpha ( \sigma _ { i b } - \sigma _ { i } ^ { 2 } ) + C _ { 2 } } ,\tag{10}
$$

where i, o, and b denote the subimages of $I , O _ { ; }$ , and $B ,$ respectively. For brevity, we omit the superscript k. We use the inverse SSIM because it is more convenient for iterative root finding.

By calculating the first and second derivatives of $\psi _ { \mu } ^ { k } ( \alpha )$ and $\psi _ { \sigma } ^ { k } ( \alpha )$ , we can show that both derivatives are positive wherever their denominators are positive. For the valid blending range $\alpha ~ \in ~ [ 0 , 1 ]$ , the denominator of $\psi _ { \mu } ^ { k }$ is always positive. The denominator of $\psi _ { \sigma } ^ { k }$ remains positive over this range when $\sigma _ { i b } ~ > ~ - C _ { 2 } / 2$ . We first consider $\sigma _ { i b } ~ > ~ - C _ { 2 } / 2$ , which is typically satisfied in practice. Under this condition, both components are positive, monotonically increasing, and convex (PMIC) over $[ 0 , 1 ] ,$ and inverse SSIM, $\psi ^ { k } ( \alpha )$ , is likewise PMIC. Because $\psi ^ { k } ( \alpha )$ is well behaved, iterative root-finding methods perform robustly. When $\psi ^ { k } ( 1 ) \geq T _ { \mathrm { S S I M } } ^ { - 1 }$ , the optimal α is 1. Otherwise, we can find the optimal α in a few iterations. Let $\alpha _ { 0 } = 1$ and $\alpha _ { 1 } < \alpha _ { 0 }$ be two initial guesses. If $\psi ^ { k } ( \alpha _ { 1 } ) > T _ { \mathrm { S S I M } } ^ { - 1 } ,$ , the secant method is expected to converge to the optimal α without oscillation. If $\psi ^ { k } ( \overset { - } { \alpha _ { 1 } } ) < T _ { \mathrm { S S I M } } ^ { - 1 } ,$ a falseposition method is more appropriate. When $\sigma _ { i b } \le - C _ { 2 } / 2$ , we set $\alpha = 0$ because the PMIC property does not hold and robust root finding becomes difficult.

We consider the noise-level dependency of $\psi _ { \mu } ^ { k } ( \alpha )$ and $\psi _ { \sigma } ^ { k } ( \alpha )$ ). Because the means are only weakly affected by noise, $\psi _ { \mu } ^ { k } ( \alpha )$ is expected to be relatively insensitive to noise level. For small subimages of noisy input, the input variance is dominated by noise and thus much larger than the output variance of the black-box denoiser. Therefore, we can expect that $\sigma _ { i } ^ { 2 } \gg \sigma _ { b } ^ { 2 }$ and $\sigma _ { i } ^ { 2 } \gg | \sigma _ { i b } |$ , and we have $\psi _ { \sigma } ^ { k } ( \alpha ) \simeq 1 + \alpha ^ { 2 } / ( 2 \bar { \alpha } )$ Note that this discussion is valid when α is away from 1, which is our main interest. Thus, $\psi _ { \sigma } ^ { k } ( \alpha )$ is also approximately insensitive to noise level. For the small-noise case, the noise dependency is not as problematic as in the large-noise case because the black-box denoiser’s output is already close to the input, and the blending process has less effect on the output. Therefore, we can expect that SSIM anchoring is more robust to noise-level variations than PSNR anchoring.

## III. NOISE DISTRIBUTION

High PSNR or SSIM between the input and output images does not guarantee that the output appears natural to human observers. In this section, we discuss naturalness from the perspective of noise distribution. Let the ground-truth (GT) image be G. The residual noise is defined as $n _ { \mathrm { r e s } } \equiv O - G$ When the distribution of $n _ { \mathrm { r e s } }$ significantly deviates from a typical Gaussian shape, the output image O is often perceived as unnatural. We use excess kurtosis $\gamma _ { 2 }$ as a simple proxy for the deviation of residual noise from the assumed Gaussian noise distribution.

Because it depends on the GT image $G ,$ which is unknown, excess kurtosis is not a practical fidelity metric. However, it is useful as a naturalness measure in controlled experiments where G is known.

## IV. EXPERIMENTS

Given an input image I and the output of a black-box denoiser $B ,$ we blend them to generate $\textit { O } = \textit { I } + \alpha \odot$ $( B - I )$ . Following the discussion in Section II, we can find the maximum feasible $\alpha$ that satisfies the fidelity condition, PSNR $( I , O ) \ \geq \ T _ { \mathrm { P S N R } }$ or $\operatorname { S S I M } ( I , O ) \geq T _ { \operatorname { S S I M } }$ , for each subimage. Within this framework, we can balance fidelity and denoising performance for a wide range of black-box denoisers, including deep learning-based methods.

In this section, we demonstrate the effectiveness of the proposed framework through experiments. As input images, we use the DIV2K validation set [3], [4], which consists of 100 high-quality images. We add zero-mean Gaussian noise with standard deviations of 25, 50, and 75 to create noisy inputs. We then apply Real-ESRGAN [5] as a black-box denoiser to obtain denoised outputs. We use Real-ESRGAN as a representative black-box restoration model because it performs both denoising and detail restoration on noisy inputs. For anchoring, we impose PSNR and SSIM constraints separately on each RGB channel: PSNR $( I _ { [ R G B ] } , O _ { [ R G B ] } ) \geq$ T<sub>PSNR</sub> and $\mathrm { S S I M } ( I _ { [ R G B ] } , { \cal O } _ { [ R G B ] } ) \geq \dot { T } _ { \mathrm { S S I M } }$ . We evaluate $T _ { \mathrm { P S N R } } ~ \in ~ \{ 1 5 , 1 \dot { 7 } . 5 , 2 0 , 2 \dot { 2 } . 5 , 2 5 , 2 7 . 5 , 3 0 \}$ and $T _ { \mathrm { S S I M } } ~ \in$ $\{ 0 . 3 , 0 . 4 , 0 . 5 , 0 . 6 , 0 . 7 , 0 . 8 , 0 . 9 \}$ . We use $7 \times 7$ windows for anchoring calculations. We also apply the OpenCV bilateral filter [6] with a conservative setting as a denoiser that produces natural-looking results but offers limited denoising performance. The filter parameters are $\mathrm { ~ d ~ } = \ 1 1$ , sigmaColor = 25, sigmaSpace = 3. While we have a closed-form solution for PSNR anchoring, SSIM anchoring requires iterative root finding, specifically the secant and false-position methods. We use a fixed number of iterations (4) for all experiments. Although 4 iterations are not necessarily optimal, they are sufficient to achieve the target SSIM values in our experiments. In the iterative process, we use $\alpha _ { 0 } = 1$ and $\alpha _ { 1 } = 0 . 7 5$ as initial guesses.

Fig. 1 reports the means and standard deviations of the achieved PSNR and SSIM values for different targets. We also use $7 \times 7$ windows to compute local SSIM values for each RGB channel and then average them to obtain the global SSIM. These results confirm that the proposed framework effectively controls denoising fidelity, showing that the locally constant assumption for α is justified to an extent and a small number of iterations are sufficient. For lower target values, the achieved values deviate from the targets because the black-box denoiser outputs already satisfy the constraints. We also show the means and standard deviations of the resulting blending factors α in Fig. 2. The blending factor α decreases as the target value increases, indicating that the output is more strongly anchored to the input image. As discussed in Section II, the blending factor $\alpha$ for PSNR anchoring is sensitive to noise level, while that for SSIM anchoring is relatively stable. This indicates that tuning the anchoring condition is easier for SSIM anchoring than for PSNR.

![](images/45b39dd52d05ce495868ee16d514d14be76417345a3e9366d6b4ded69918d119.jpg)

![](images/585e1d6edce83fc018feb4290cc55bf879a47e13bc3a4bbbeffbc29fbb92912a.jpg)  
Fig. 1. Achieved PSNR (left) and SSIM (right) values for different target values. (Real-ESRGAN)

![](images/3af77930907c8bed0a70d604cae1df8a3020cbd0048764e68d8b91c162230225.jpg)

![](images/44319225286b9807ad9c786b510e665d5f83b902cca93eca71d84efd972a9ae9.jpg)  
Fig. 2. Blending factor α vs. anchor PSNR (left) and anchor SSIM (right). (Real-ESRGAN)

We evaluate denoising performance using PSNR between the GT image G and the output image O. We use PSNR rather than SSIM because SSIM between GT and denoised outputs is often very low (e.g., 0.1 or 0.2), making cross-method comparisons less informative. In contrast, PSNR provides a straightforward pixel-wise fidelity measure. While it is a convenient metric, PSNR alone does not guarantee perceptual naturalness. Many deep learning-based denoisers achieve high PSNR but often produce unnatural artifacts. To investigate the perceptual naturalness of noise, we analyze the excess kurtosis of residual noise, $n _ { \mathrm { r e s } } \ = \ O - G ,$ as discussed in Section III. For evaluation, PSNR is computed across the RGB channels, whereas the excess kurtosis $\gamma _ { 2 }$ is evaluated on the luma (grayscale) residual noise, because human perception of noise is primarily governed by luma artifacts.

Fig. 3 (top: $\sigma = 2 5$ , middle: $\sigma = 5 0$ , bottom: $\sigma = 7 5 )$ shows PSNR and excess kurtosis for different target values<sup>1</sup>. These results indicate that fidelity control provides a practical way to balance denoising performance and statistical naturalness. For $\sigma = 2 5$ PSNR anchoring even achieves higher PSNR than Real-ESRGAN; we leave detailed analysis of this behavior for future work. The two anchoring strategies exhibit clearly different trends. Across $\sigma = 2 5 , \sigma = 5 0 \ /$ , and $\sigma = 7 5$ , SSIM anchoring shows similar PSNR-kurtosis trade-off curves, and $T _ { \mathrm { S S I M } } ~ = ~ 0 . 8$ is a reasonable operating point for all three noise levels, yielding sufficiently high PSNR (compared to bilateral) and low excess kurtosis. In contrast, PSNR anchoring behaves differently across noise levels. Although $T _ { \mathrm { P S N R } } = 2 5$ yields low excess kurtosis in all cases, the achieved PSNR is relatively high for $\sigma = 2 5$ but low for $\sigma = 5 0$ and $\sigma = 7 5 .$ Thus, PSNR anchoring is sensitive to noise level, while SSIM anchoring is relatively stable. This noise dependency of PSNR anchoring is consistent with the previous analysis.

![](images/83a50b853add39ce91c439b3dca8f52d0b5071a87f40351dfedc924a82a23e54.jpg)  
Fig. 3. PSNR and excess kurtosis of the residual noise for different target values under Gaussian noise with σ = 25 (top), σ = 50 (middle), and $\sigma = 7 5$ (bottom). Real-ESRGAN is used as a black-box denoiser.

To further validate these findings, we conduct additional experiments using the non-local means denoiser [7] as another black-box denoiser. We use the OpenCV implementation, fastNlMeansDenoisingColored, with parameters $\mathrm { ~ h ~ } = \ 2 5$ , hColor = 25, templateWindowSize = 7, and searchWindowSize = 21. Fig. 4 shows the anchoring results for different conditions, indicating that PSNR and SSIM anchoring also work well for non-local means denoising. Fig. 5 (top: $\sigma = 2 5$ , bottom: $\sigma = 5 0 )$ shows the corresponding PSNR and excess kurtosis of the residual noise. We report only the $\sigma = 2 5$ and $\sigma = 5 0$ cases to save space. The same trend is observed: SSIM anchoring exhibits more consistent behavior across noise levels than PSNR anchoring, suggesting that this behavior is not specific to Real-ESRGAN. Thus, tuning the anchoring condition is easier for SSIM anchoring than for PSNR.

## V. CONCLUSION

In this paper, we proposed a framework that anchors the output of a black-box denoiser to the input image using fidelity metrics. We demonstrated that local blending between denoiser output and input image enables explicit control of fidelity metrics such as PSNR and SSIM. Experiments showed that this approach effectively balances denoising performance and statistical naturalness, measured by excess kurtosis of residual noise. In particular, SSIM anchoring exhibits more consistent behavior across noise levels, making it a robust choice for practical applications. The framework treats the denoiser as a black-box mapping and does not require modification or retraining.

![](images/615158341356ef6dc26c0805c81cf1f6420b3761a63a8da90db1bb7d7068838f.jpg)

![](images/0dd632d9da0132c3c64ca515c66763ba1ace7ba02a7bd7a7b0a2eb09aee3619b.jpg)  
Fig. 4. Achieved PSNR (left) and SSIM (right) values for different target values. (Non-local Means)

![](images/2df1612748ed9d97f32472ecf172230140cd1e2a1e0476e7ebb26697b93ee00a.jpg)  
Fig. 5. PSNR and excess kurtosis of the residual noise for different target values under Gaussian noise with $\sigma = 2 5$ (top) and $\sigma = 5 0$ (bottom). Nonlocal means is used as a black-box denoiser.

Future work includes exploring additional fidelity metrics and metric combinations, and extending the framework to more complex noise models and real-world scenarios. Importantly, the proposed strategy of blending black-box outputs with inputs under explicit fidelity constraints is generic and can be extended to other image restoration tasks, such as superresolution and deblurring.

A limitation of our framework is that, because it is based on linear blending, performance degrades when the denoiser output differs substantially from the input or when the input noise is so strong that the original image structure is heavily corrupted.

## ACKNOWLEDGMENT

The author would like to thank Takeshi Miura for insightful discussions, and Masaki Hilaga for his support and authorization to conduct this research as part of corporate duties. An AI language model was used to assist with the manuscript’s structure and English phrasing.

## REFERENCES

[1] C. Morikawa, M. Kobayashi, M. Satoh, Y. Kuroda, T. Inomata, H. Matsuo, T. Miura, and M. Hilaga, “Image and video processing on mobile devices: a survey,” The Visual Computer, vol. 37, pp. 2931–2949, 2021.

[2] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.

[3] E. Agustsson and R. Timofte, “Ntire 2017 challenge on single image super-resolution: Dataset and study,” in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, July 2017.

[4] R. Timofte, E. Agustsson, L. Van Gool, M.-H. Yang, L. Zhang, B. Lim et al., “Ntire 2017 challenge on single image super-resolution: Methods and results,” in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, July 2017.

[5] X. Wang, L. Xie, C. Dong, and Y. Shan, “Real-esrgan: Training realworld blind super-resolution with pure synthetic data,” in International Conference on Computer Vision Workshops (ICCVW).

[6] C. Tomasi and R. Manduchi, “Bilateral filtering for gray and color images,” in Proceedings of the sixth IEEE international conference on computer vision. IEEE, 1998, pp. 839–846.

[7] A. Buades, B. Coll, and J.-M. Morel, “A non-local algorithm for image denoising,” in 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), vol. 2. IEEE, 2005, pp. 60–65.