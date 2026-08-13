# Evaluating and Calibrating Difusion Model-derived Uncertainty for Quantitative MRI Mapping

Shishuai Wang<sup>1</sup>, Stefan Klein<sup>1</sup>, Juan A. Hernandez-Tamames<sup>1,2</sup>, and Dirk H.J. Poot<sup>1</sup>

<sup>1</sup> Department of Radiology and Nuclear Medicine, Erasmus MC, Rotterdam, The Netherlands

2 Department of Imaging Physics, TU Delft, Delft, The Netherlands s.wang@erasmusmc.nl

Abstract. Quantitative MRI (qMRI) provides standardised tissue parameter maps, but the reliability of deep learning-based qMRI mapping methods is often not explicitly characterised. In this work we systematically evaluate uncertainty maps for quantitative MRI derived from multiple inferences of a data-consistent difusion model-based qMRI framework. Evaluation on synthetic test data assessed error-awareness, higherror detection, selective prediction, and Gaussian interval calibration. Difusion model-derived uncertainty was positively associated with the mapping error, while risk-coverage analysis showed that excluding highuncertainty voxels reduced the retained error. However, the raw uncertainty was poorly calibrated for quantitative interval interpretation. Calibration was substantially improved using a post-hoc procedure combining prediction-value-dependent bias correction with scalar uncertainty scaling. Qualitative evaluation on a healthy volunteer showed spatially meaningful uncertainty patterns. These results indicate that difusion model-derived uncertainty is informative for reliability assessment and selective prediction, but requires calibration for quantitative interval interpretation.

Keywords: Uncertainty estimation · Difusion models · Quantitative MRI.

## 1 Introduction

Unlike conventional MRI, where image contrast is determined by a mixture of tissue properties and acquisition settings, quantitative MRI (qMRI) explicitly estimates biophysical tissue parameters such as proton density and relaxation times. The resulting quantitative maps ofer a more standardised representation of tissue properties, which is desirable for cross-subject analysis, multi-centre studies, and longitudinal monitoring. Quantitative MRI typically requires the acquisition of multiple contrast-weighted images with varying sequence parameters, from which quantitative maps are subsequently estimated. Conventional estimation approaches rely on signal models, such as non-linear model fitting or dictionary matching, while recent deep learning-based methods have shown promising performance. However, uncertainty estimation, which is important for assessing the reliability of the resulting quantitative maps, is often not routinely provided or systematically evaluated.

For deep learning-based methods, epistemic uncertainty can be estimated using approaches such as Monte Carlo dropout [4] or model ensembles [5, 6, 10, 8]. Nevertheless, these methods typically require architectural modifications, additional model training, or specialized inference strategies. In contrast, diffusion models have recently shown promise for qMRI mapping [7, 1, 9]. Due to their stochastic sampling process, difusion models can produce diferent predictions from the same input, making the standard deviation from repeated inference results a straightforward way to estimate uncertainty without additional model modifications or retraining [2]. Previous work has preliminarily shown that the difusion model-derived uncertainty visually overlaps with error-prone regions [7]. However, it remains unclear whether such uncertainty estimates are quantitatively associated with mapping errors, whether they can support selective prediction, and whether they are calibrated when interpreted as Gaussian prediction intervals.

In this work, we systematically evaluate difusion model-derived uncertainty for quantitative T1 and T2 mapping using a data-consistent difusion-based qMRI framework. Mean and standard deviation of repeated inferences are used as output quantitative map and uncertainty map respectively. The evaluation is performed on synthetic data with available ground truth, and an additional real healthy volunteer scan used for qualitative visualisation. To summarise, our contributions are:

We provide a systematic evaluation of difusion model-derived uncertainty in qMRI, covering error-awareness, high-error detection, selective prediction, and coverage calibration.

– We show that sampling uncertainty is informative for reliability assessment and uncertainty-guided selective prediction, but is not inherently calibrated as a Gaussian prediction interval.

– We investigate a simple post-hoc calibration strategy based on predictionvalue-dependent bias correction and scalar uncertainty scaling, which improves Gaussian interval calibration on the test dataset.

## 2 Materials and Methods

## 2.1 Difusion model-derived uncertainty

A difusion model-based qMRI mapping framework [9] is used as a testbed in this study. The model is based on a denoising difusion probabilistic model integrated with data-consistency enforcement during inference. The input to the model is the stacked multi-echo weighted images. For each input, the stochastic sampling process was repeated K times, yielding predictions $\mathsf { \hat { y } } ^ { ( 1 ) } , \dots , \hat { \pmb { y } } ^ { ( K ) }$ . For each parameter $p \in \{ T _ { 1 } , T _ { 2 } \}$ , the mean prediction and difusion model-derived uncertainty are defined as

$$
\mu _ { p } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \hat { y } _ { p } ^ { ( k ) } , \mathrm { ~ a n d ~ } \sigma _ { p } = \sqrt { \frac { 1 } { K - 1 } \sum _ { k = 1 } ^ { K } \left( \hat { y } _ { p } ^ { ( k ) } - \mu _ { p } \right) ^ { 2 } }\tag{1}
$$

respectively. Given the ground-truth map $y _ { p }$ , the absolute error map and standardised residual are defined as

$$
e _ { p } : = | y _ { p } - \mu _ { p } | , { \mathrm { ~ a n d ~ } } r _ { p } : = { \frac { y _ { p } - \mu _ { p } } { \sigma _ { p } } }\tag{2}
$$

respectively. Voxels with $\sigma _ { p } \leq 1 0 ^ { - 8 }$ were excluded from the standardised residual and calibration correction for numerical stability. All operations are applied voxel-wise.

## 2.2 Experimental data

Training, calibration and test datasets were generated from 20 BrainWeb digital brain phantoms [3]. Nineteen phantoms were used for model training and posthoc calibration, while the remaining phantom was held out for testing. For model training, five independently generated quantitative-map realisations were created for each of the 19 training phantoms. For post-hoc calibration, one additional realisation was generated for each of these 19 phantoms and these realisations were not used for model training. For testing, 20 independently generated realisations were created from the held-out phantom. For each realisation of quantitative maps, noisy k-space data were generated and the corresponding weighted images were reconstructed. The model training and inference hyperparameters were kept the same as [9]. During inference, the selected qMRI framework was applied K = 10 times to compute $\mu _ { p }$ and $\sigma _ { p }$

In addition, one previously acquired healthy volunteer scan using the same $\mathrm { \Phi { q M R I } }$ protocol was included for qualitative visualisation. The scan was acquired on a 3.0T GE SIGNA Premier scanner using a 48-channel head coil. Since voxelwise ground-truth quantitative maps are unavailable in vivo, this scan was not included in the quantitative evaluation.

## 2.3 Evaluation metrics

All metrics were computed within the foreground head mask and reported for each parameter as mean ± standard deviation across the test dataset. Mapping accuracy was evaluated using the mean absolute error (MAE) and root mean squared error (RMSE) between $\mu _ { p }$ and $y _ { p }$ . The magnitude of difusion modelderived uncertainty was summarised using the mean and 95th percentile of $\sigma _ { p }$

To assess error-awareness, we computed the Spearman correlation between $e _ { p }$ and $\sigma _ { p } .$ . As an indication of high-error detection performance, we computed the area under the receiver operating characteristic curve (AUROC) when using $\sigma _ { p }$ to detect the 10% voxels with the highest $e _ { p }$

Selective prediction was evaluated using risk-coverage analysis. Voxels were sorted from low to high uncertainty. For a retained coverage fraction $c ,$ only the c fraction of voxels with the lowest uncertainty was retained, and the retained risk was computed as the MAE over these voxels. The area under the risk-coverage curve (AURC) was used as a summary measure, where lower values indicate better selective prediction. For comparison, we also computed a random-ranking baseline, representing chance-level voxel selection, and an oracle-error-ranking baseline, where voxels were sorted from low to high true absolute error.

Calibration was evaluated by interpreting the uncertainty estimate as a Gaussian predictive standard deviation. For a nominal central coverage level $q \in ( 0 , 1 )$ 2 the corresponding Gaussian interval was defined using

$$
z _ { q } = \varPhi ^ { - 1 } \left( \frac { 1 + q } { 2 } \right) ,\tag{3}
$$

where $\varPhi$ denotes the standard normal cumulative distribution function. The empirical coverage probability was computed as

$$
C P ( q ) = \frac { 1 } { | \varOmega | } \sum _ { x \in \varOmega } \mathbf { 1 } ( | y _ { p } ( x ) - \mu _ { p } ( x ) | \leq z _ { q } \sigma _ { p } ( x ) ) .\tag{4}
$$

where $\varOmega$ denotes the foreground mask and $\mathbf { 1 } ( \cdot )$ is the indicator function. The coverage probability accuracy at nominal coverage level $q$ was defined as

$$
C P A ( q ) = | C P ( q ) - q | ,\tag{5}
$$

with lower values indicating better interval calibration. We used the central 50% interval as the main scalar calibration metric, denoted as $C P A _ { 5 0 }$ for brevity, corresponding to $q = 0 . 5$ and $z _ { 0 . 5 } = \varPhi ^ { - 1 } ( 0 . 7 5 ) = 0 . 6 7 4$ . Calibration curves were additionally evaluated over multiple nominal coverage levels.

## 2.4 Post-hoc calibration

To investigate whether miscalibration could be mitigated, we applied a simple post-hoc calibration procedure using the calibration dataset. For each parameter p, a prediction-value-dependent bias correction was first estimated. Calibration voxels were binned according to their predicted value $\mu _ { p }$ using 50 quantile bins, and the median residual $y _ { p } - \mu _ { p }$ was computed within each bin. This yielded an empirical lookup table $\hat { b } _ { p } ( \mu _ { p } )$ , which was linearly interpolated and applied to correct the mean prediction:

$$
\begin{array} { r } { \mu _ { p , \mathrm { c o r r } } = \mu _ { p } + \hat { b } _ { p } ( \mu _ { p } ) . } \end{array}\tag{6}
$$

Predictions outside the calibrated range were assigned the nearest boundary correction value.

After bias correction, a scalar uncertainty scaling factor was estimated to improve central interval calibration. For the nominal central 50% Gaussian interval, the scaling factor was estimated as

$$
\lambda _ { p } = \frac { \mathrm { m e d i a n } \left( | y _ { p } - \mu _ { p , \mathrm { c o r r } } | / \sigma _ { p } \right) } { z _ { 0 . 5 } } ,\tag{7}
$$

where the median was computed over foreground calibration voxels. The calibrated uncertainty was then defined as

$$
\sigma _ { p , \mathrm { c o r r } } = \lambda _ { p } \sigma _ { p } .\tag{8}
$$

The lookup table $\hat { b } _ { p } ( \mu _ { p } )$ and scaling factor $\lambda _ { p }$ were estimated on the calibration dataset and then fixed when applied to the held-out test dataset. On the test dataset, calibration was evaluated for three variants: the raw prediction, the bias-corrected prediction, and the bias-corrected prediction with scaled uncertainty. The same fixed $\hat { b } _ { p } ( \mu _ { p } )$ and $\lambda _ { p }$ were used for all nominal coverage levels, and no additional calibration was performed for individual values of $q .$

## 3 Results

A representative synthetic data example is shown in Fig. 1. Elevated uncertainty was predominantly observed around tissue interfaces and in regions exhibiting increased absolute mapping error. Table 1 summarizes the quantitative evaluation results. Difusion model-derived uncertainty was positively associated with mapping errors. The uncertainty-error Spearman correlation was 0.260 ± 0.095 for T1 and $0 . 6 1 2 \pm 0 . 1 1 3$ for T2. The representative scatter plots in Fig. 2 further illustrate this association. Uncertainty also detected the highest-error voxels, yielding AUROC values of 0.773±0.076 and 0.953±0.010, respectively. As shown in Fig. 3, excluding high-uncertainty voxels reduced the retained MAE, showing that difusion model-derived uncertainty can support uncertainty-guided selective prediction. Compared with the baselines, the uncertainty-guided curves were consistently below random ranking and above oracle-error ranking for both T1 and T2. This indicates that the uncertainty estimates provide useful, although non-optimal, voxel-wise reliability ranking.

Table 1. Summary of Uncertainty Evaluation on Synthetic Test Data
<table><tr><td>Parameter</td><td>MAE↓</td><td>RMSE↓</td><td>Mean Uncertainty p95 Uncertainty</td><td></td></tr><tr><td>T1 (s)</td><td>0.110±0.029</td><td>0.148±0.034</td><td>0.059±0.003</td><td>0.166±0.010</td></tr><tr><td>T2 (s)</td><td>0.030±0.007</td><td>0.054±0.017</td><td>0.017±0.002</td><td>0.077±0.008</td></tr><tr><td>Parameter U-E Spearman</td><td></td><td>AUROC ↑ 7</td><td>AURC↓</td><td>Raw CPA50 ↓</td></tr><tr><td>T1</td><td>0.260±0.095</td><td>0.773±0.076</td><td>0.079±0.022</td><td>0.287±0.064</td></tr><tr><td>T2</td><td>0.612±0.113</td><td>0.953±0.010</td><td>0.012±0.001</td><td>0.370±0.033</td></tr></table>

![](images/4d4dc3c811a9a7ba5dbe3b05a13dc8bd019d9aeaa45f7f941facf09cc0adad90.jpg)  
Fig. 1. Representative qualitative results on synthetic test data. The unit is seconds.

Despite its error-awareness and selective utility, the raw uncertainty showed poor interval calibration. The raw $C P A _ { 5 0 }$ was $0 . 2 8 7 \pm 0 . 0 6 4$ for T1 and $0 . 3 7 0 \pm$ 0.033 for T2, indicating substantial deviations from the nominal 50% coverage. As summarized in Table 2, the prediction-value-dependent bias correction reduced the MAE from $0 . 1 1 0 \pm 0 . 0 2 9 \mathrm { ~ s ~ }$ to $0 . 0 8 9 \pm 0 . 0 1 8 \mathrm { ~ s ~ }$ for T1, and from $0 . 0 3 0 { \pm } 0 . 0 0 7 \ \mathrm { s }$ to $0 . 0 2 3 { \pm } 0 . 0 0 6 \ \mathrm { s }$ for T2. It also reduced $C P A _ { 5 0 }$ to 0.228 ± 0.044 and $0 . 2 4 0 \pm 0 . 0 3 9$ , respectively. Subsequent uncertainty scaling using $\lambda _ { p }$ further reduced $C P A _ { 5 0 }$ to $0 . 0 6 3 { \pm } 0 . 0 5 4$ for T1 and $0 . 0 5 5 { \pm } 0 . 0 4 2$ for T2. Figure 4 visualizes the standardised residual $( r _ { p } )$ distributions for one representative case. The raw residuals were shifted and substantially broader than the standard normal distribution. Bias correction reduced the central ofset, while uncertainty scaling further improved the nominal 50% interval coverage. Nevertheless, heavy-tailed residuals remained, particularly for T2, indicating that the post-hoc procedure improved central coverage calibration without making the complete residual distribution Gaussian.

![](images/7b1afd74908b2b7c9c710e3732d0d8900947a0f71b2f21ef9ef6289aa5f0e55c.jpg)

![](images/6ec4757b8355e936f95ec31ddf3c1ce7c5bc416ce436581965da511b410f6558.jpg)  
Fig. 2. Voxel-wise association between absolute mapping error and difusion modelderived uncertainty for one representative case. Spearman correlations were computed using all valid foreground voxels, while points were randomly subsampled for visualisation.

![](images/f0bea600c997c85180b2aa2e18087c3705e843335660793608f386c1af435ef6.jpg)

![](images/c9157accf4ecf54a4e8a72aa16c3bdbf09b512049f8728abb8846d6fab9b43fa.jpg)  
Fig. 3. Risk-coverage analysis for uncertainty-guided selective prediction. Voxels were retained from low to high difusion model-derived uncertainty. Random ranking represents chance-level selection, while oracle ranking retains voxels from low to high true absolute error and therefore provides an unattainable lower bound. Solid lines and shaded regions indicate mean ± standard deviation across 20 synthetic test realisations.

Table 2. Post-hoc Calibration Correction Summary
<table><tr><td>Parameter</td><td>Raw MAE↓</td><td>Bias-corr. MAE↓</td><td>Raw  $C P A _ { 5 0 } \downarrow$ </td><td></td><td>Bias-corr.  $C P A _ { 5 0 } \downarrow$ </td><td>Bias+scale-corr.  $C P A _ { 5 0 } ~ .$  1</td></tr><tr><td>T1</td><td></td><td>1.750 0.110±0.029 0.089±0.018 0.287±0.064 0.228±0.044</td><td></td><td></td><td></td><td> $\overline { { 0 . 0 6 3 { \pm } 0 . 0 5 4 } }$ </td></tr><tr><td>T2</td><td></td><td>1.837 0.030±0.007 0.023±0.006 0.370±0.033 0.240±0.039</td><td></td><td></td><td></td><td> $0 . 0 5 5 { \pm } 0 . 0 4 2$ </td></tr></table>

The calibration curves in Fig.5 further confirmed this behaviour across multiple nominal coverage levels. The raw intervals under-covered across the evaluated range, bias correction improved empirical coverage, and subsequent scalar uncertainty scaling shifted the curves closer to the ideal diagonal, particularly around the central coverage range targeted by the scaling procedure. Residual undercoverage at higher nominal coverage levels indicates that the post-hoc correction improved central interval calibration but did not fully calibrate the predictive distribution.

Finally, Fig. 6 shows qualitative results on in vivo data. Dictionary matching was included as a conventional model-based reference for qualitative comparison, but was not treated as ground truth. The uncertainty maps showed elevated values mainly around tissue interfaces, and within CSF and grey matter. The real-data example was used for qualitative illustration only, as ground truth was unavailable.

![](images/490366ea15e68ce0bdcb477663bf0b0b23c0c20698a416731d160bda78663a7d.jpg)

Fig. 4. Standardised residual distributions before and after post-hoc calibration for one representative case. The standard normal density is shown for reference, and the dashed vertical lines indicate the central 50% Gaussian interval.  
![](images/62f57902f605af51ddf930c88b9434934d102f2e8e6555f8bb6d58e5d98040b6.jpg)  
Fig. 5. Calibration curves over multiple nominal coverage levels. The lookup-table bias correction and scalar uncertainty scaling were estimated on the calibration set and then fixed when applied to the test data. The scalar scaling factor was estimated to match the central 50% interval. Lines and shaded regions indicate mean ± standard deviation across 20 test realisations.

## 4 Discussion

In this work, we treated the standard deviation across repeated difusion model samples as difusion model-derived uncertainty and systematically evaluated it for quantitative MRI mapping. The results show that this uncertainty contains meaningful information about prediction reliability: it is positively associated with mapping errors, detected high-error regions, and supported uncertaintyguided selective prediction. However, it was not inherently calibrated when interpreted as a Gaussian predictive standard deviation. A simple post-hoc procedure combining prediction-value-dependent bias correction and scalar uncertainty scaling substantially improved the calibration on test data.

The diferent evaluation metrics capture complementary aspects of uncertainty quality. AUROC evaluates whether uncertainty ranks voxels in the tail of the error distribution above lower-error voxels. The strong high-error detection performance, especially for T2, could be used to flag or reject highly unreliable estimation in practice. The positive uncertainty-error correlations further indicate a continuous association between uncertainty and error magnitude, while the risk-coverage results demonstrate that this ranking can be used to reduce retained error by excluding uncertain voxels.

![](images/4a8f0691fcc4ed3326939e417d0aa7d919e94d87298ac4ad9630a28f91acebad.jpg)  
Fig. 6. Qualitative results on a healthy volunteer scan. The unit is seconds.

Despite this error-awareness, the raw uncertainty substantially under-covered the nominal 50% Gaussian prediction intervals. This indicates that such uncertainty estimation may be useful for ranking predictions without having the correct absolute scale for interval estimation. The standardised residual distributions were shifted and broader than a standard normal distribution, indicating contributions from both systematic prediction bias and underestimated uncertainty magnitude. The prediction-value-dependent lookup table reduced systematic bias and improved the accuracy of the mean quantitative maps. Subsequent scalar uncertainty scaling corrected the remaining scale mismatch and markedly reduced $C P A _ { 5 0 }$ on the test data. However, the post-hoc correction should not be interpreted as making the full predictive distribution Gaussian. Heavy-tailed standardised residuals remained, particularly for T2. The proposed scaling was derived specifically to match the nominal 50% interval and therefore calibrates the central part of the residual distribution rather than its extremes.

Several limitations should be considered. Quantitative evaluation was limited to synthetic data, as ground truth is unavailable for in-vivo qMRI. Although the calibration dataset used newly generated quantitative-map realisations, the underlying anatomies had been seen by the model during training. The uncertainty estimate was computed from $K = 1 0$ repeated difusion model inferences, which was chosen as a practical compromise between computational cost and repeated-sampling stability. The sensitivity of the reported uncertainty metrics to K was not systematically investigated. Moreover, this sampling-based uncertainty may not capture all sources of predictive uncertainty, including model bias and acquisition-related domain shift. The healthy-volunteer experiment therefore provides qualitative evidence only. Validation on additional anatomies, acquisition protocols, and pathological cases is required in future work.

Overall, difusion model-derived sampling uncertainty was informative for identifying unreliable qMRI estimations and enabling selective prediction, but its raw magnitude should not be interpreted directly as a calibrated Gaussian predictive standard deviation. The results suggest that simple post-hoc bias and scale correction can substantially improve central interval calibration while preserving the practical error-awareness of the uncertainty estimate.

Acknowledgments. This work was conducted within the “Trustworthy AI for MRI” ICAI lab within the project ROBUST, funded by the Dutch Research Council (NWO), GE Healthcare, and the Dutch Ministry of Economic Afairs and Climate Policy (EZK).

Disclosure of Interests. Shishuai Wang, Stefan Klein, Juan-Antonio Hernandez-Tamames and Dirk Poot received research grants from GE Healthcare.

## References

1. Bian, W., Jang, A., Zhang, L., Yang, X., Stewart, Z., Liu, F.: Difusion modeling with domain-conditioned prior guidance for accelerated MRI and qMRI reconstruction. IEEE Transactions on Medical Imaging (2024)

2. Chung, H., Ye, J.C.: Score-based difusion models for accelerated MRI. Medical image analysis 80, 102479 (2022)

3. Cocosco, C.A.: Brainweb: Online interface to a 3D MRI simulated brain database. NeuroImage p. 425 (1997)

4. Gal, Y., Ghahramani, Z.: Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In: Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48. p. 1050–1059. ICML’16, JMLR.org (2016)

5. Lakshminarayanan, B., Pritzel, A., Blundell, C.: Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems 30 (2017)

6. Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J., Lakshminarayanan, B., Snoek, J.: Can you trust your model’s uncertainty? Evaluating predictive uncertainty under dataset shift. Advances in neural information processing systems 32 (2019)

7. Wang, S., Ma, H., Hernandez-Tamames, J.A., Klein, S., Poot, D.H.: qMRI diffuser: quantitative T1 mapping of the brain using a denoising difusion probabilistic model. In: MICCAI Workshop on Deep Generative Models. pp. 129–138. Springer (2024)

8. Wang, S., Nuyts, J., Filipovic, M.: Uncertainty estimation in liver tumor segmentation using the posterior bootstrap. In: International workshop on uncertainty for safe utilization of machine learning in medical imaging. pp. 188–197. Springer (2023)

9. Wang, S., Wiesinger, F., Sgambelluri, N., Pirkl, C., Klein, S., Hernandez-Tamames, J.A., Poot, D.H.: q3-MuPa: Quick, quiet, quantitative multi-parametric MRI using physics-informed difusion models. Magnetic Resonance Imaging p. 110699 (2026)

10. Zhao, Y., Yang, C., Schweidtmann, A., Tao, Q.: Eficient Bayesian uncertainty estimation for nnU-Net. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 535–544. Springer (2022)