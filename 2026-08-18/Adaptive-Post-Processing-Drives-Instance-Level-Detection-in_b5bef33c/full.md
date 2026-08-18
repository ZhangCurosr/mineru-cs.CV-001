# Adaptive Post-Processing Drives Instance-Level Detection in Stroke Lesion Segmentation

Qinghui Liu<sup>1⋆</sup>, Jon André Ottesen<sup>1</sup>, Atle Bjørnerud<sup>1</sup>, and Kyrre Eeg Emblem<sup>1</sup>

Oslo University Hospital, Norway

qiliu@ous-hf.no

Abstract. Instance-level lesion detection has been an increasingly larger focal point in medical image segmentation besides the more standard voxel-level overlap. Still, most pipelines are trained and post-processed for voxel overlap alone. In particular, the mismatch is most pronounced for small lesions, where a near-miss prediction—substantial overlap that falls just short of the instance-matching threshold—scores the same as a complete miss. In our ISLES’26 submission, we found that closing this gap mattered far more in post-processing than in architecture design. Our Volume-Conditioned Adaptive Post-Processing (VCAP) scheme adjusts component-size thresholds to each case’s predicted lesion burden, improving Lesion-F1 by 0.032 (unbiased cross-fold estimate), approximately ∼6× larger than any architectural change we tested. A resolution-aware attention architecture (Viola2Plus), designed for small-lesion segmentation, shows why the distinction matters: it left small-lesion Dice unchanged but raised small-lesion detection rate by 3.7%, a real efect voxeloverlap metrics alone would have missed. Under 5-fold cross-validation on the 1,453-case training set, our post-processed two-architecture ensemble achieves Dice 0.651 and Lesion-F1 0.614, versus 0.644 and 0.573 for the unprocessed single-model baseline.

Keywords: Stroke lesion segmentation · ISLES’26 · Instance-level evaluation · Post-processing · Viola Attention U-Net.

## 1 Introduction

Segmentation models are almost always trained to maximize voxel overlap between expert-based ground-truth annotations and model predictions, most commonly via Dice and cross-entropy losses. Computed over large volumes, these losses are dominated by large lesions – a single 300 ml lesion contributes far more gradient signal than a 0.05 ml lacunar lesion. This training-time bias becomes a problem once a model is judged not just on how much it overlaps the ground truth, but on instance-level criteria such as lesion counts and total burden: a loss that barely notices small lesions produces a model that is correspondingly bad at finding them. The ISLES’26 challenge [3, 5] scores submissions across five metrics: Dice, instance-matched Lesion-F1 (IoU ≥ 0.25), Absolute Lesion-count Diference (ALD), Absolute Volume Diference (AVD), and PR-AUC.

![](images/ef44ffe93f200fcb39024360791f72ca5b464c4692ea381edd69744fb2fe8354.jpg)  
Fig. 1: The instance-matching gate. (A) Axial crop: ground-truth (green) and predicted (red) contours overlap substantially along a thin, ventricle-hugging lesion. (B) GT is 37.1 ml, prediction is 10.0 ml, overlap is 8.9 ml – yet the best instance-level IoU is only 0.209. (C) Because 0.209 falls short of the instancelevel matching threshold $\mathrm { ( I o U \ge 0 . 2 5 ) }$ , no instance pair qualifies and $\mathrm { t p } = 0 \colon$ the naive voxel Dice this overlap would “deserve” is 0.380, but the competitionscored Dice and Lesion-F1 are both forced to 0.

Confirmed at the evaluation-library level [3], Dice and Lesion-F1 share the same instance-matching gate: if no predicted component reaches IoU ≥ 0.25 with any ground-truth lesion, Dice is forced to zero no matter how many voxels overlap (Figure 1). Training on voxel-level Dice does nothing about this edge case on its own.

To address the challenges described above, our paper focuses on reducing the gap between the predicted lesion-burden and the scored lesion burden. Postprocessing calibrated to predicted lesion burden is the most efective approach we found for closing the aforementioned problem, and the resulting configuration transfers across architectures without retraining. A stratified failure analysis also shows that an attention architecture can raise small-lesion detection without changing voxel overlap – a reminder that voxel-level and instance-level gains are not interchangeable evidence of a method’s performance.

## 2 Methods

## 2.1 Dataset and Preprocessing

The ISLES’26 training set comprises 1,453 multi-center T1-weighted MRI cases from 72 acquisition cohorts (71 sites plus the SOOP cohort). Figure 2 summarizes the cohort’s heterogeneity: acquisition orientation, voxel-spacing anisotropy, lesion volume, and metadata completeness all vary substantially – time since stroke onset, for instance, is missing for 23.5% of cases. Intensity scale also varies by up to three orders of magnitude across centers, which is why we normalize per case rather than per cohort. All models use images resampled to isotropic 1 mm spacing.

![](images/214de4b96b264dfef34060347897cfeeefc6644fb0232d861a7fa911a082cd1b.jpg)

![](images/c4e570635e9834cf458ae378a2231dc99ecfee075a8023ddfc560b5228f6b3a7.jpg)

![](images/0eb4084681727edbc99e0de0e61065aef47abd6094165a3e613ffcaf6059e7ad.jpg)

![](images/3803b233e124671897092fe87fc9af711e0a17e1cc93544dbc718257af5fae7c.jpg)  
Fig. 2: Dataset heterogeneity (n = 1453). (A) Acquisition orientation: predominantly RAS (59.8%) and LAS (39.6%). (B) Anisotropy: most scans are near-isotropic (<1.25 mm), with 7.3% moderately to severely anisotropic. (C) Lesion volume is heavy-tailed (median 5.2 ml), including 5 empty ground-truth cases. (D) Time since stroke onset, missing for 342 cases.

## 2.2 Network Architecture

We compare two architecture families under identical plans geometry, data, and fold partitions (128<sup>3</sup> patch, batch size 2, deep supervision, 1 mm isotropic), so architecture is the only varied factor:

1. Baseline: a stock nnU-Net PlainConvUNet [2, 9].

2. Viola2Plus: deep decoder stages (320/256/128 channels) use tri-axial global attention over pooled axial statistics, in the spirit of Viola-UNet [6, 7]. Shallow, high-resolution stages (64/32 channels) apply a local spatial gate in the spirit of the attention gates introduced for medical segmentation by Oktay et al. [8], modulated by upsampled deep features.

All attention multipliers are initialized to the identity, so training starts from the plain baseline and only learns deviations where the data supports them. The local gate is boost-only, its multiplier bounded to [1, 2], so it can amplify a region but never suppress one (Section 3.2). Both models use the standard nnU-Net compound loss (Dice plus cross-entropy).

## 2.3 Volume-Conditioned Adaptive Post-Processing (VCAP)

A single global threshold and fixed minimum-component-size filter cannot handle lesion burdens spanning several orders of magnitude: a permissive filter (0.02 ml) preserves lacunar infarcts but retains false positives around large lesions, while a strict filter (0.1 ml) cleans up large infarcts yet discards small true positives. VCAP resolves this by making the component-size threshold a function of the case’s own predicted burden.

Algorithm 1 Volume-Conditioned Adaptive Post-Processing (VCAP)   
Require: soft map $p \in [ 0 , 1 ] ^ { \Omega } ;$ ; voxel volume v (ml); binarization threshold $\theta \ =$   
0.35; burden tiers $( V _ { 1 } , V _ { 2 } ) \ : = \ : ( 3 . 5 , 3 5 )$ ml; per-tier minimum component volumes   
$( \tau _ { 1 } , \tau _ { 2 } , \tau _ { 3 } ) = ( 0 . 0 2 , 0 . 0 2 , 0 . 1 )$ ml; empty rule $\tau _ { \mathrm { e m p t y } } = 0 . 0 2$ ml   
Ensure: binary mask $b ;$ calibrated soft map $\tilde { p }$   
1: $b \gets \mathcal { H } [ p \geq \theta ]$ ▷ recall-first binarization (lowered from 0.5)   
2: $V   \boldsymbol { b }  \cdot \boldsymbol { v }$ ▷ total predicted burden for this case   
3: if $V < V _ { 1 }$ then $\tau  \tau _ { 1 }$   
4: else if $V \leq V _ { 2 }$ then $\tau  \tau _ { 2 }$   
5: else $\tau  \tau _ { 3 }$   
6: end if   
7: $\{ C _ { 1 } , \ldots , C _ { K } \} $ ConnectedComponents(b, 26-connectivity)   
8: $\begin{array} { r } { b  \bigcup _ { k : | C _ { k } | \cdot v > \tau } C _ { k } } \end{array}$ ▷ volume-conditioned component filter   
9: if $| b | \cdot v \leq$ τ<sub>empty</sub> then   
10: $b  \mathbf { 0 } ;$ $\tilde { p } \gets \mathbf { 0 }$ ▷ empty rescue (empty-GT PR-AUC edge case)   
11: else   
12: $\tilde { p }  p$   
13: end if   
14: return $b , \ \tilde { p }$

Specifically, cases with predicted burden below 35 ml use a permissive 0.02 ml filter to retain small lesions, while cases above 35 ml switch to a stricter 0.1 ml filter to suppress noisy fragments around large, heterogeneous infarcts. A nearempty rule zeroes out any prediction whose total retained volume falls below 0.02 ml, targeting the empty-ground-truth edge case under PR-AUC. We deliberately lowered the binarization threshold from 0.5 to 0.35 to push undersegmented lesions past the $\mathrm { I o U } \geq 0 . 2 5$ matching gate. All thresholds were jointly optimized via grid search on pooled out-of-fold predictions, balancing all five metrics. Connected components are computed with cc3d [10] at 26-connectivity. The complete rule is stated compactly in Algorithm 1.

## 3 Results

Oficial test-set labels are not available before submission, so all reported metrics come from 5-fold cross-validation on the training set, pooled across folds. The instance-matching gate $\mathrm { ( I o U \ge 0 . 2 5 ) }$ is reproduced using the organizers’ panoptica package [4, 1]. Table 1 reports cross-validation performance before post-processing or ensembling. Adoption of Viola2Plus followed a rule set before we saw any results: adopt only if fold-0 Dice improved by at least 0.005 and Lesion-F1 by at least 0.01. Fold 0 cleared both bars (+0.0105 Dice, +0.0212 F1), and the remaining folds confirm the direction – pooled over all five, Viola2Plus improves every metric (Dice +0.0045, F1 +0.0057, PR-AUC +0.0048, AVD −0.23 ml, ALD −0.02).

Table 1: 5-fold cross-validation results, before post-processing. Baseline (Base) = stock nnU-Net PlainConvUNet; Viola2Plus (V2+) = dual-stage attention; identical data, folds, and plans geometry.
<table><tr><td rowspan="2">Fold</td><td colspan="2">Dice ↑</td><td colspan="2">AVD (ml) ↓</td><td colspan="2">ALD↓</td><td colspan="2">Lesion-F1 ↑</td><td colspan="2">PR-AUC ↑</td></tr><tr><td>Base</td><td>V2+</td><td>Base</td><td>V2+</td><td>Base V2+</td><td></td><td>Base</td><td>V2+</td><td>Base</td><td>V2+</td></tr><tr><td>0</td><td>0.6569</td><td>0.6674 5.20</td><td></td><td>5.30</td><td>1.72</td><td>1.75</td><td>0.5811</td><td>0.6023</td><td>0.7546 0.7651</td><td></td></tr><tr><td>1</td><td>0.6638</td><td>0.6636</td><td>6.87</td><td>6.36</td><td></td><td>2.39 2.28 0.5879</td><td></td><td>0.5873</td><td>0.7633 0.7652</td><td></td></tr><tr><td>2</td><td>0.6231 0.6225</td><td></td><td>4.95</td><td>4.60</td><td></td><td>1.81 1.81 0.5766</td><td></td><td>0.5677</td><td>0.72100.7203</td><td></td></tr><tr><td>3</td><td>0.6427 0.6516 5.91</td><td></td><td></td><td>5.86</td><td></td><td>1.73 1.77 0.5578 0.5652</td><td></td><td></td><td>0.7442 0.7556</td><td></td></tr><tr><td>4</td><td>0.63430.63854.82</td><td></td><td></td><td>4.50</td><td>1.901.83</td><td></td><td>0.5608</td><td>0.5699</td><td>0.7364 0.7373</td><td></td></tr><tr><td colspan="9">Pooled0.6442 0.6487 5.55 5.32 1.91 1.89 90.5728</td><td>0.5785 0.74390.7487</td></tr></table>

![](images/39c7a39bd74b4aec1d47b9d8fe3547b7a66cb2d4a193ec3bf86700424c629209.jpg)  
Fig. 3: Efect of VCAP on a thick-slice case. Left: raw prediction (threshold 0.5) with spurious fragments (yellow dotted). Center: after VCAP, fragments removed (17 → 5 components). Right: per-case metrics (Lesion-F1 0.261 → 0.667, ALD 14 → 2) alongside pooled 5-fold deltas.

Table 2: Post-processing ablation (pooled 5-fold, n = 1453). VCAP + = VCAP adds the near-empty zero-out rule. Ensemble = per-case average of baseline and Viola2Plus soft maps. The configuration was tuned on the baseline and applied unchanged to the others.
<table><tr><td>Configuration Dice ↑ AVD</td><td colspan="4">(ml) ↓ALD. ↓Lesion-F1 ↑ PR-AUC ↑</td></tr><tr><td>Baseline</td><td>0.6442</td><td>5.55</td><td>1.91</td><td>0.5728</td><td>0.7439</td></tr><tr><td>Viola2Plus</td><td>0.6487</td><td>5.32</td><td>1.89</td><td>0.5785</td><td>0.7487</td></tr><tr><td>Ensemble</td><td>0.6488</td><td>5.37</td><td>1.85</td><td>0.5831</td><td>0.7556</td></tr><tr><td>Baseline, VCAP+</td><td>0.6461</td><td>5.62</td><td>1.83</td><td>0.6095</td><td>0.7512</td></tr><tr><td>Viola2Plus, VCAP+</td><td>0.6494</td><td>5.40</td><td>1.81</td><td>0.6122</td><td>0.7534</td></tr><tr><td>Ensemble, VCAP+</td><td>0.6509</td><td>5.52</td><td>1.82</td><td>0.6143</td><td>0.7599</td></tr></table>

![](images/6ef015f16406b56d073f8e760d792f13a6077bb022edd6d9c1ab0573fc7f919d.jpg)  
Strata definitions: T1a (<0.05ml, n=16) | T1b (0.05-0.1ml, n=27) | T2 (0.1-0.5ml, n=145)  
Fig. 4: Performance on small-lesion strata (pooled 5-fold, no postprocessing). Lesions are categorized by ground-truth volume: T1a (<0.05 ml), T1b (0.05–0.1 ml), and T2 (0.1–0.5 ml). While Viola2Plus demonstrates a clear advantage in instance-level retrieval—consistently boosting (b) Detection Rate across all tiers and improving (c) PR-AUC for lesions > 0.05 ml—it yields no corresponding benefit in voxel-level overlap, as evidenced by the stagnant or slightly degraded (a) Dice scores. This highlights the architecture’s capacity to amplify weak signals for detection without refining spatial boundaries.

## 3.1 VCAP Yields the Greatest Instance-Level Gains

Figure 3 shows this on one case: VCAP removes spurious components, taking Lesion-F1 from 0.261 to 0.667 and ALD from 14 to 2. Re-tuning the grid search on the Viola2Plus and ensemble outputs shifts the optimal binarization threshold from 0.35 to 0.5, but gains at most 0.002 on any metric over the original configuration – a wide, flat optimum – so we kept one configuration for all three families (Table 2).

VCAP improves cohort-level Lesion-F1 by 0.037 over the unfiltered baseline (0.5728 → 0.6095) under full-data selection; under an unbiased leave-one-foldout nested-selection protocol, the estimate is 0.032, stable across folds – the number we treat as the honest efect size. The gain holds across all three model families, though it shrinks slightly as the underlying model improves: VCAP adds +0.0367 F1 to the baseline, +0.0337 to Viola2Plus, and +0.0312 to the ensemble, consistent with post-processing and model quality partially overlapping in what they fix. This is not free: AVD rises by 0.07 ml relative to the unfiltered baseline, the cost of lowering the binarization threshold to rescue under-segmented lesions past the matching gate.

## 3.2 Architectural Nuance: Enhancing Detection, Not Overlap

Viola2Plus was designed to specifically improve small-lesion segmentation, but its empirical behavior reveals a more nuanced dynamic (Figure 4). Across the 188 small-lesion cases (<0.5 ml), the overall detection rate rises by 3.7 points (0.713 to 0.750)—about three times the architecture’s cohort-wide detection improvement.

When stratified by volume, Viola2Plus consistently increases instance-level detection across all tiers, with the most pronounced gain in the extremely challenging T1a tier (<0.05 ml, +6.3% points). Notably, the precision-recall AUC (PR-AUC) also improves for T1b and T2 lesions, indicating that this heightened sensitivity does not come at the severe cost of precision. However, this enhanced retrieval ability does not translate to better voxel-level overlap: smalllesion Dice scores remain flat or slightly decrease (overall −0.003, p = 0.25), and AVD gets marginally worse (+0.19 ml).

This behavioral pattern is highly consistent with the local gate’s boost-only design. A mechanism that can amplify a candidate region but never suppress one efectively pushes borderline small lesions past the detection threshold— improving both detection rate and PR-AUC—but it lacks the inherent capability to tighten their spatial boundaries, resulting in stagnant Dice scores and the observed AVD increase.

## 4 Discussion and Conclusion

Our volume-conditioned adaptive post-processing (VCAP) approach produced an instance-level gain roughly 6 times larger than any architecture change we tried (0.032 vs. 0.006 Lesion-F1). The two levers are largely independent: gains from ensembling, post-processing, and architecture stack rather than compete. A single global threshold implicitly assumes a scale-invariant trade-of between keeping true positives and rejecting false ones, which breaks down whenever lesion size spans several orders of magnitude – common in stroke imaging and probably elsewhere. Once calibrated, VCAP transfers across model families with negligible loss.

The small-lesion result makes the same point from another angle: an architecture built to improve overlap instead shifted detection upward while leaving overlap flat – something a Dice-only evaluation would have read as no efect at all. Wherever Dice and an instance-matching metric share a detection gate, a change can do something real that overlap metrics alone cannot see, so architecture claims under instance-aware metrics are worth checking against detection rate and volume bias directly, not inferred from Dice.

Two limitations are relevant for these conclusions. The empty-ground-truth subgroup is small (5 of 1,453 cases) and cannot support a strong claim either way. VCAP’s gain is also not free: AVD rises by 0.07 ml, the cost of lowering the binarization threshold to rescue borderline lesions. Our 5-fold internal estimate, finally, is not an oficial test-set result.

Acknowledgments. The authors acknowledge support from the Helse Sør-Øst regional health authority of Norway (Grant 2021031).

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

[1] Hernandez Petzsche, M.R., de la Rosa, E., Hanning, U., Wiest, R., Valenzuela Pinilla, W., Reyes, M., Meyer, M.I., Liew, S.L., Kofler, F., Ezhov, I., Robben, D., Hutton, A., Friedrich, T., Zarth, T., Bürkle, J., Baran, T.A., Menze, B., Broocks, G., Meyer, L., Zimmer, C., Boeckh-Behrens, T., Berndt, M., Ikenberg, B., Wiestler, B., Kirschke, J.S.: ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset. Scientific Data 9, 762 (2022). https://doi.org/10.1038/ s41597-022-01875-5

[2] Isensee, F., Jaeger, P.F., Kohl, S.A.A., Petersen, J., Maier-Hein, K.H.: nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods 18, 203–211 (2021). https://doi.org/10. 1038/s41592-020-01008-z

[3] ISLES’26 Challenge Organizers: ISLES’26: Ischemic stroke lesion segmentation challenge 2026 – evaluation framework. GitHub repository (2026), https://github.com/ezequieldlrosa/isles26, accessed: 2026-08-04

[4] Kofler, F., Möller, H., Buchner, J.A., de la Rosa, E., Ezhov, I., Rosier, M., Mekki, I., Shit, S., Negwer, M., Al-Maskari, R., Ertürk, A., Vinayahalingam, S., Isensee, F., Pati, S., Rueckert, D., Kirschke, J.S., Ehrlich, S.K., Reinke, A., Menze, B., Wiestler, B., Piraud, M.: Panoptica – instance-wise evaluation of 3D semantic and instance segmentation maps (2023)

[5] Liew, S.L., Lo, B., Donnelly, M.R., Zavaliangos-Petropulu, A., Jeong, J.N., Barisano, G., Hutton, A., Simon, J.P., Juliano, J.M., Suri, A., et al.: A large, curated, open-source stroke neuroimaging dataset to improve lesion segmentation algorithms. Scientific Data 9(1), 321 (2022). https://doi.org/10. 1038/s41597-022-01401-7, https://fcon\_1000.projects.nitrc.org/ indi/retro/atlas.html

[6] Liu, Q., MacIntosh, B.J., Schellhorn, T., Skogen, K., Emblem, K., Bjørnerud, A.: Voxels intersecting along orthogonal levels attention u-net for intracerebral haemorrhage segmentation in head ct. In: Proceedings of ISBI 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI) (2023)

[7] Liu, Q., Nesvold, J.E., Raaum, H., Murugesu, E., Røvang, M., MacIntosh, B.J., Bjørnerud, A., Skogen, K.: Examining deployment and refinement of the viola-ai intracranial hemorrhage model using an interactive neomedsys platform. BMC Methods 3(1), 30 (Aug 2026). https://doi.org/10.1186/ s44330-026-00083-6, https://doi.org/10.1186/s44330-026-00083-6

[8] Oktay, O., Schlemper, J., Le Folgoc, L., Lee, M., Heinrich, M., Misawa, K., Mori, K., McDonagh, S., Hammerla, N.Y., Kainz, B., Glocker, B., Rueckert, D.: Attention U-Net: Learning where to look for the pancreas. In: Medical Imaging with Deep Learning (MIDL) (2018)

[9] Ronneberger, O., Fischer, P., Brox, T.: U-Net: Convolutional networks for biomedical image segmentation. In: Medical Image Computing and

Computer-Assisted Intervention (MICCAI). LNCS, vol. 9351, pp. 234–241. Springer (2015). https://doi.org/10.1007/978-3-319-24574-4\_28

[10] Silversmith, W.: cc3d: Connected components on multilabel 3D & 2D images (2021). https://doi.org/10.5281/zenodo.5535251, zenodo release v3.2.1