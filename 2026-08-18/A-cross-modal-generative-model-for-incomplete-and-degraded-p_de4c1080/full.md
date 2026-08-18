# A cross-modal generative model for incomplete and degraded prostate MRI with multicentre clinical validation

Siyuan Ma<sup>1,†</sup>, Liang He<sup>2,†</sup>, Mengying Zhu<sup>3,†</sup>, Yi Chai<sup>4,†</sup>, Mengyao Lyu<sup>1,†</sup>, Haowei Wang<sup>5,†</sup>, Qizhen Lan<sup>6,†</sup>, HaoBo Sun<sup>7</sup>, Qixin Zhang<sup>1</sup>, Jingli Chen<sup>3</sup>, Xiaobing Wei<sup>3</sup>, Jiaming Liu<sup>8</sup>, Guiqin Liu<sup>3</sup>, Qianwen Zhang<sup>9</sup>, Yang Liu<sup>1,∗</sup>, Dacheng Tao<sup>1,∗</sup>, Guangyu Wu<sup>3,∗</sup>

<sup>1</sup>College of Computing and Data Science, Nanyang Technological University, Singapore <sup>2</sup>School of Computer Science and Technology, Tongji University, Shanghai, China <sup>3</sup>Department of Radiology, Ren Ji Hospital, Shanghai Jiao Tong University School of Medicine, Shanghai, China <sup>4</sup>National University of Singapore, Singapore

<sup>5</sup>Department of Medical Oncology, Shanghai East Hospital, Tongji University School of Medicine, Shanghai, China <sup>6</sup>McWilliams School of Biomedical Informatics, The University of Texas Health Science Center at Houston (UTHealth Houston), Houston, Texas, USA

<sup>7</sup>Jiangsu Key Laboratory of Urban ITS, Jiangsu Province Collaborative Innovation Center of Modern Urban Trafic Technologies, School of Transportation, Southeast University, Nanjing 210096, China <sup>8</sup>Department of Radiology, Ningbo Hangzhou Bay Hospital, Ningbo, Zhejiang, China <sup>9</sup>Department of Radiology, Changhai Hospital, Naval Medical University, Shanghai, China

<sup>†</sup>These authors contributed equally to this work.

<sup>∗</sup>Co-corresponding authors: Yang Liu, Dacheng Tao and Guangyu Wu. Correspondence e-mail: danielrau@163.com (G.W.)

## Abstract

Missing or degraded sequences can limit prostate multiparametric MRI. We developed MSCNet, a sequence-conditioned cross-modal generative framework for reconstructing unavailable contrasts and restoring degraded acquisitions. Across ten completion tasks, task-specific MSCNet achieved mean structural similarity of 0.818 versus 0.798 for the strongest task-matched comparators; matched-capacity analyses showed larger diferences in lesion fidelity and boundary preservation. In a blinded 1,000-case reader study, overall image quality met the prespecified non-inferiority criterion for DWI, ADC and T2W completion, but not T1W. In a separate 200-case diagnostic assessment, AUCs for clinically significant cancer were 0.860 with acquired images, 0.841 with MSCNet and 0.797 with baseline-generated images. A locked 186-case three-hospital cohort supported multicentre transportability. These retrospective results support quality-controlled cross-modal reconstruction as an adjunct to acquired prostate MRI.

Multiparametric magnetic resonance imaging (mpMRI) has changed the diagnostic pathway for suspected prostate cancer by improving detection of clinically significant disease and reducing unnecessary biopsy. 1–4 Its value arises from the joint interpretation of sequence-specific evidence. T2-weighted imaging (T2W) depicts zonal anatomy, gland margins and tumour morphology, whereas difusion-weighted imaging (DWI) and apparent-difusion-coeficient (ADC) maps characterize water mobility and cellularity.<sup>1;5</sup> T1-weighted imaging (T1W) provides complementary information on haemorrhage and pelvic background. The Prostate Imaging Reporting and Data System (PI-RADS) formalizes this sequence-dependent interpretation.<sup>1</sup> The diagnostic integrity of an examination therefore depends not only on the presence of disease, but also on whether the required contrasts are complete, spatially aligned and interpretable.

Sequence quality remains variable in routine practice. DWI is particularly vulnerable to susceptibilityinduced distortion, signal dropout, motion and ghosting, and these efects can become more conspicuous at higher field strengths despite gains in signal-to-noise ratio. <sup>5;6</sup> Motion, patient preparation, scanner hardware and protocol heterogeneity can also degrade T2W or make an otherwise complete examination locally non-diagnostic. PI-QUAL and PI-QUAL version 2 were developed because image quality directly afects lesion detection, confidence in a negative examination and downstream biopsy planning.<sup>7;8</sup> In referral workflows, where protocols and sequence availability vary, an incomplete or degraded contrast can therefore create a practical choice between interpreting an uncertain examination and repeating the scan.

Cross-modal synthesis ofers a route to estimate unavailable information from the remaining sequences. Earlier work established modality-invariant latent representations, hetero-modal learning, multi-input adversarial synthesis and hybrid or unified fusion strategies for missing-modality imputation.<sup>11–15</sup> Generic adversarial and image-translation models can generate visually plausible outputs,<sup>18;19;37</sup> but perceptual sharpness alone does not establish anatomical or diagnostic fidelity. Most previous evaluations have focused on brain MRI, fixed input configurations or image-level similarity. Prostate MRI poses a stricter problem: a clinically useful system must handle diferent available-sequence combinations, preserve target-specific anatomy and lesion contrast, remain stable under acquisition artefacts, expose uncertainty and be assessed against radiologist and diagnostic endpoints.

We developed MSCNet as a sequence-conditioned reconstruction framework that preserves modalityspecific representations, fuses available contrasts through target-conditioned gates and combines global-context modelling with attention-guided decoding and edge-aware supervision. Evaluation separated physically coupled DWI–ADC tasks from cross-contrast synthesis and used capacity- and objective-matched controls, modern generative baselines, a shared-routing variant, physics-aware controls and retrieval. Clinical relevance was assessed through anatomical and lesion-level endpoints, same-patient repeat scans, a blinded reader study, cancer discrimination, a locked three-hospital external cohort, failure adjudication and selective prediction.

## Results

## Study design and a cross-modal reconstruction framework

The study addressed two related failure modes of prostate mpMRI (Fig. 1). In multi-sequence completion, one target sequence was treated as unavailable and reconstructed from the remaining sequences. In virtual enhancement, an acquired sequence afected by artefact or global degradation was restored using complementary information from the other contrasts. The public and institutional cohorts supported three 2-to-1 tasks in T2W, DWI and ADC, and four 3-to-1 tasks after inclusion of T1W. The principal analysis trained one model for each target configuration; a secondary shared model used a target token and modality mask to process all ten configurations with one set of weights. A clinical artefact cohort, 204 same-patient repeat acquisitions, a 1,000-case reader study, a 200-case diagnostic assessment and a pooled 186-case three-hospital external cohort (site A, 72 cases; site B, 64; site C, 50) extended evaluation beyond image similarity (Extended Data Table 1).

MSCNet was designed around the observation that MRI contrasts are not interchangeable image channels. Separate encoders first retained modality-specific representations. At each feature scale, target-conditioned gates then controlled how much information from each available sequence entered the fused skip representation. A Transformer bottleneck modelled long-range context,<sup>21</sup> and an attention-gated decoder<sup>22</sup> reconstructed the target while suppressing features that were irrelevant or dominated by artefact. Squeeze-and-excitation blocks <sup>23</sup> and deep supervision<sup>24</sup> supported channel recalibration and stable multiscale optimization. Ablation analysis showed that the full model outperformed removal or replacement of each component across distortion, structural and perceptual metrics; the largest aggregate degradation followed removal of the global-context bottleneck, while gated fusion and the edge-aware branch contributed complementary gains (Supplementary Fig. 2). The ablation pattern assigned complementary roles to global anatomical context, target-conditioned cross-modal routing and boundary-sensitive supervision.

Data governance was treated as part of the modelling problem. Splits were defined at the patient level, and sequence pairs underwent orientation checks, anatomical matching and deformable correction before inclusion. Registration correction reduced landmark errors across sequence pairs and anatomical levels, whereas ADC-DWI pairs remained the most intrinsically aligned. The quality-control pipeline excluded 272 cases, most commonly for missing sequences, severe artefact or sequence mismatch (Supplementary Fig. 4); detailed cohort governance and exclusion criteria are reported in (Supplementary Table 1). PROSTATEx served as a vendor-specific public-cohort analysis and was excluded from the independent external-validation

![](images/16fbc7e22001c75a6c03318b58b903409320ec8839665ffebab95f7eabc044f2.jpg)  
Figure 1. Study overview, cohorts, completion tasks and MSCNet framework. A, Clinical motivation and workflow for incomplete or degraded prostate mpMRI, cross-modal completion or enhancement, and radiologist review. B, Public, institutional, artefact and reader-study cohorts. C, Matrix of 2-to-1 and 3-to-1 completion tasks. D, MSCNet architecture, synthesis outputs and optimization objectives, including modality-specific encoders, multi-scale cross-modal gated fusion, global-context modelling, attention-gated decoding, deep supervision and edge-aware reconstruction.

## Reconstruction gains occurred in anatomical and lesion-relevant regions

Representative cases showed distinct failure patterns among the comparison methods (Fig. 2A). Regression baselines produced smooth gland interiors and softened capsule margins; adversarial baselines recovered sharper texture but introduced local signal instability. MSCNet reduced both behaviours. The residual maps were less spatially extensive, and magnified regions retained gland contours and intraprostatic structure across

T1W, T2W, DWI and ADC targets.

The regional analyses established that these gains were not driven by uniform background. Absolute residuals were lowest for MSCNet within the gland and boundary ring, where errors are most likely to afect lesion conspicuity or zonal assessment (Fig. 2B). The advantage persisted at the apex, mid-gland and base (Fig. 2C). In the local volumetric analysis, adjacent-slice consistency was 0.942 for MSCNet, compared with 0.908 for DynUNet and 0.874 for the GAN baseline; slice-to-slice variation was correspondingly lower (0.018 versus 0.031 and 0.050). Volumetric smoothness and z-trend correlation showed the same ordering (Fig. 2D). Boundary-specific measures, including edge preservation, boundary SSIM, gradient similarity and capsule sharpness, were also highest for MSCNet (Fig. 2E). Together, these results indicate that improved similarity reflected preservation of coherent anatomy rather than only denoising of large low-information regions.

Lesion-centred analyses provided a second test of anatomical fidelity. Correlations between acquired and generated lesion contrast were 0.959 for contrast-to-noise ratio, 0.962 for lesion-region signal-to-noise ratio and 0.966 for lesion-to-background contrast ratio, compared with 0.792, 0.821 and 0.853 for Pix2Pix (Supplementary Fig. 5). Line profiles showed that MSCNet followed acquired lesion peaks, troughs and transition zones more closely than the comparison methods. Radiologist-rated lesion conspicuity was 4.29 for MSCNet and 4.47 for acquired reference images, compared with 3.62 for the difusion baseline, 3.33 for the residual-transformer baseline and 2.73 for Pix2Pix (Supplementary Fig. 5). The remaining gap from acquired images was concentrated in small lesions, uncertain boundaries and dificult T2W reconstruction, which became important in the subsequent safety analysis.

（A)  
![](images/fc799fb6f2f07c67565008af265911a25eae7a1eccdcc82c5a115382928ed230.jpg)

（B)  
![](images/0b11e1d78a87fa4dd57f6d14db29d5e5b40a07240ba020e29a85c30424479da1.jpg)

（C)  
![](images/9453e77526517caa0c697e834df0369f088ebc51ee55b2a0c36c95dfa1b9fd00.jpg)

![](images/837ae62adc458d75cece6ed9b72369022a8885b07c6581a4b6e4e4cc8a2d6ab0.jpg)

![](images/adcd8c1418a0a8edc57e52976f1d6ed255d6861564502f3e5cd0523c11b3684f.jpg)

（D)  
![](images/8ae6371fe7db3d3854e1584a941a5471d3fbcab0ba21483b7e10b86d3892881b.jpg)

![](images/d4fbde2308cd4cad4eeaa38fd24869c92558f3dd92f54a4ae8faea62624d7873.jpg)

（E)  
![](images/fb960dcec7ded3da5738a6edfa7fbb875c715a116ee28e3b44ea6e381280a2d0.jpg)

![](images/1935289a09ef37ac6d4d941d257a9858cbe2a0f769ea00037319caeec2b200ff.jpg)

![](images/d3f67c6b97e0896e080e722942e4cb951ffe0bdeb4d64b44836bfbbf8718704c.jpg)

![](images/20b22868b0fb13cebc0cdbe9889630ea9bcf6d044cb8947ff65a97b86851f315.jpg)

![](images/cbd6a78d14318a99c74b5c148df3ba7786c0800eecbf163d2ba7329d5f733cee.jpg)

![](images/bc03ad052deaa2be84e7e26a47208f74af702c10f1cef18666a4a2c4aaabefaa.jpg)  
Figure 2. Visual and spatial behaviour of multi-sequence prostate MRI completion. A, Input sequences, acquired reference targets, baseline outputs, MSCNet outputs, residual maps and enlarged regions. B, Residual decomposition in the gland, boundary ring and peri-gland regions. $\mathbf { C } ,$ Local structural similarity, residual error and boundary error at the apex, mid-gland and base. D, Adjacent-slice consistency, slice-to-slice variation, volumetric smoothness and z-trend correlation. $\mathbf { E } ,$ Edge preservation, boundary SSIM, gradient similarity and capsule sharpness.

## Performance across targets, cohorts and acquisition strata

MSCNet achieved the highest SSIM in all ten completion tasks after expansion of the comparison set to unified synthesis, parameter-matched Transformer, masked-difusion, structure-aware latent-difusion, loss-matched DynUNet and retrieval or signal-model controls (Fig. 3A–C and Extended Data Table 2). Because ADC and DWI share difusion-derived information, we prespecified three task groups rather than interpreting every task as equally dificult. Mean SSIM was 0.867 for physically coupled DWI–ADC tasks, 0.727 for cross-contrast T2W tasks and 0.796 for T1W completion. Relative to the strongest non-MSCNet model selected within each task, the corresponding gains were 0.018, 0.024 and 0.020, respectively (Supplementary Table 14). The ten-task macro-average was 0.818 versus 0.798 for the strongest task-wise comparators, an absolute gain of 0.020. The largest group-level advantage occurred in the cross-contrast tasks, where the target could not be recovered from a direct difusion signal relationship.

Matched controls separated the contribution of model capacity, optimization and sequence-conditioned routing. Relative to the parameter-matched Transformer on the locked subset, MSCNet increased lesion fidelity from 0.87 to 0.92 and boundary SSIM from 0.853 to 0.908 while reducing LPIPS from 0.108 to 0.086 (Fig. 3E and (Supplementary Table 11)). These localized diferences provide a clinical-anatomical anchor for the smaller whole-image SSIM efect without treating SSIM as a diagnostic surrogate. Prespecified module ablation yielded aggregate SSIM values of 0.931 for the full model, 0.917 without the edge-aware branch, 0.912 without gated fusion, 0.906 with direct concatenation and 0.899 without the Transformer bottleneck (Supplementary Table 11). Target-specific optimized signal-model and retrieval controls also remained below MSCNet on their applicable held-out tasks (Supplementary Table 15).

A secondary shared variant tested unified input–target routing with one set of weights. MSCNet-Shared achieved mean SSIM 0.804, compared with 0.818 for optimized task-specific MSCNet, while reducing storage from ten 154.1-million-parameter models to one 158.3-million-parameter model (Supplementary Table 12). Unified pretraining on 3,750 development cases followed by task-specific fine-tuning reached SSIM 0.818 and LPIPS 0.108, whereas few-shot, frozen-encoder and unseen-combination performance remained lower (Supplementary Table 13). The shared model therefore provides a unified routing extension across evaluated configurations rather than evidence of unrestricted zero-shot behaviour.

Generality was examined across public, institutional, temporal and hospital-level settings (Fig. 3F and (Supplementary Fig. 3)). Correlations across scanner and protocol strata ranged from 0.86 to 0.94. In the pooled three-hospital external cohort, the macro-average across tasks A–C was SSIM 0.791, LPIPS 0.124 and MAE 0.061, compared with 0.809, 0.113 and 0.056 in the internal held-out cohort (Supplementary Table 16). This shift supports transportability of the reconstruction signal while leaving prospective workflow validation and hospital-specific monitoring necessary.

![](images/ecff79b4a2e19628f83591c2a341788dd45caed7970cd13a0603629f0376d8cf.jpg)

![](images/4b956d1235a8bd7ebb8cc103bbc2d782dc3e4d679d65f4937450b94dc0472148.jpg)  
(E)

(C)  
![](images/b606c699f8d3689a92ce601e851fb6569de157df284ec7a7a7b9daa6d2743b0a.jpg)

![](images/cbb3391af8b68b000fc0039442c8ad649e9924cb926c1107c0dc24b0258ad948.jpg)

![](images/c5f52a7011e5ac88560b35475d8424bba76d0b043d80af80d2fa2da79b89e9e2.jpg)  
(F)

![](images/3db62edf0a871ca3018daf36223fd70e1bb8cf6dbd9f40638491ec7e583a1596.jpg)

![](images/7fc45d598728e9a8e15a3c40291cae3a0e449731ce08bd5cb59ef5102b7fd667.jpg)

![](images/eb65066ab6d0dbc6082694a31f2e7df7774006e44a466074e1be4d8c2d7a857d.jpg)

![](images/106f1dc09013f8bbf3d2214d80e3fa10057d6a243ebc4869b3a53b222c1efd51.jpg)

![](images/3a97718a76b6b91bded3fc8742c6c8abf4ebb509aa40a8526d70aa342d8d284b.jpg)

(I)  
![](images/0acf1219c3ad171601d710f207f2608df627713ee793e13e2bba20a504b2d30c.jpg)

![](images/0284b54e1fceb2be484b33e61276150cfb7c4f3f7c16207a59f1c7d1d58ede5d.jpg)

(J)  
![](images/15c5a0a36a2d7cc26f99d6a7a7eff9889c697eba165031944709f5851fae598c.jpg)

![](images/eed3544bc58564e4cd2a35808d58a418658d9270d151066fcd37e01cfb3e6e2b.jpg)

![](images/98db20dbabcf9fa601d3598feb8fba2f8c05ba28884797ada2248173f5d8d22f.jpg)

![](images/66c154c758592792d5c7e523f8bb3c802a3008fa0cd8f6f983add918eb9b1991.jpg)

![](images/781131a93fcf92fb1e000b048564d5698631da0ab36597c1df2dae3a83291640.jpg)  
Figure 3. Optimized quantitative performance and clinical-anatomical efect size. A, Task-level SSIM for MSCNet and the strongest task-specific non-MSCNet comparator. B, Absolute SSIM gain for each completion task. C, Grouped macro SSIM for physical-coupling, cross-contrast and T1W tasks. D, Task-level LPIPS for MSCNet and the strongest task-specific comparator. E, Matched-capacity analysis on the locked subset, showing whole-image SSIM, lesion fidelity and boundary SSIM. F, Internal and external-hospital SSIM, including the three external sites and pooled estimate.

## Clinical artefact transfer extended the model from completion to enhancement

Missing sequences are only one manifestation of incomplete information. In practice, a sequence may be present but locally unreliable. We therefore extracted clinically observed artefact patterns from the institutional archive and transferred them to quality-controlled targets under anatomical and fidelity constraints (Fig. 4A). The procedure covered susceptibility artefact, motion ghosting, banding, Gibbs ringing and chemical shift, rather than only generic noise or blur. A strict pairing step used lesion-boundary checks and an SSIM threshold of 0.8 to reject implausible degraded-reference pairs.

Enhancement improved every artefact category (Fig. 4B,C). Across the cohort, PSNR increased from 22.7 to 31.6, SSIM from 0.338 to 0.654 and MS-SSIM from 0.368 to 0.684. The higher-is-better transformations of LPIPS, MAE and FID increased from 0.464, 0.414 and 0.316 to 0.736, 0.670 and 0.628, respectively. PSNR gains were observed for susceptibility artefact (21.4 to 29.9), motion ghosting (22.6 to 30.5), banding (19.8 to 31.7), Gibbs ringing (25.2 to 33.3) and chemical shift (24.5 to 32.6). Concordance among the six quality measures and consistent separation from the comparison models indicated that the efect was not attributable to a single metric (Fig. 4D,E and (Supplementary Table 3)).

The safety analysis changed the interpretation of these improvements. Among radiologist-adjudicated outputs, any safety event occurred in 9.4% of MSCNet outputs and 28.7% of baseline outputs. Suspected false lesions occurred in 1.4% versus 5.8%, reduced lesion conspicuity in 3.6% versus 11.2%, blurred lesion margins in 5.1% versus 15.6%, partial lesion erasure in 2.3% versus 8.4%, and non-diagnostic outputs in 3.2% versus 9.3% (Supplementary Fig. 1). Severe motion and susceptibility artefacts were predominantly acquisition-driven, whereas very small lesions were predominantly lesion-driven and sequence misregistration was predominantly registration/model-driven. Thus, enhancement reduced risk but did not remove it; the residual events were structured and potentially detectable. The adjudicated event rates and their operational definitions are summarized in (Supplementary Table 7).

A separate same-patient repeat-scan analysis tested whether the controlled-transfer result extended to genuine degraded acquisitions. The 182-patient artefact cohort yielded 204 paired acquisitions because one patient could contribute more than one adjudicated degradation stratum. Mean SSIM increased from 0.714 for the degraded acquisition to 0.883 after MSCNet restoration when compared with a clean repeat scan, an absolute increase of 0.169 (� < 0.001 with patient clustering). Improvements were consistent across motion blur, susceptibility distortion, low-SNR high-b-value DWI, geometric distortion, zipper artefact, ghosting and radiofrequency inhomogeneity (Supplementary Table 17). This analysis complements rather than replaces the controlled artefact-transfer experiment: the repeat scans provide a clinically acquired reference, whereas the controlled pairs isolate specific corruption classes.

(E)  
(A)  
![](images/33f36f3c87a229aac5f4e14a7d8a258979de9435a8a0ca370298c5d135091a74.jpg)

![](images/b38db0dcdf55c25d0fd9fef4b0ebe755ad52ab790a7a2b2a73652f4c0930f082.jpg)

![](images/44d622ee7a3b999df7ecfe7e00862e3bc663f74554c0b1b83316619e64d056ef.jpg)

![](images/829c26ab3f302eeefb76d22d229b0ac4bc337635b3d0ee7fddf7f12af6d9b279.jpg)

![](images/7baa58453df85af41fa074c8d53ac6b24ca1b44093dc6ed100d9e98a6d980c8c.jpg)

![](images/c1e2a788d9d2041237d3d887f2d7e1fca9e786bb6f9dc6fc9ab73ba32e5b6ff9.jpg)

![](images/e93e0c0d3e353b8d3cd7e2398917e8d522a54a86a4587b21ff7a5e0a55a031ae.jpg)

![](images/b15ef05e4d89103627d5567cf6da7b3edb61e5b5b4ec711fae8e5178eb13b0e1.jpg)

![](images/99bb988c7524a1c26f439750655e6b8da2dd2419a111e1bed3075339958200e3.jpg)

![](images/3fc4f6bbe45a08cb9ad282a68058e1fa539f20042d137261e6e9f82e53bbb6a9.jpg)

![](images/0df57a60313f8b3991fc66df7ae55dd1a906d8056848617cfcb2d3d42e50851e.jpg)

![](images/cda7ec2e1cbfcf19f664e04cbdfebe9fc385e254410cff015791070eba857be1.jpg)

![](images/b50e2f2af89b20938f8dce78769fe0f0c77e7ca8c6f85d1e8a7486ece85c3d08.jpg)

![](images/6e0cc9a01a384fe2bb599a9f8af927c578b4651b12a4ef45d9064def5ff3e51e.jpg)

![](images/519766b21fc366306e979b30c28bc1dfa700e3f93b7899f95365759095637be9.jpg)

![](images/2caac840de32dcb7e3a1cd14513204813a0df926a9676f03ae3a9a86e4dd6b78.jpg)

![](images/880a4e5482d91cebcf5cfbef52cb9a57286ff70973f4103febfd313510bc86a9.jpg)

![](images/a08d9ee73e285949ef581c3dda8298fc3da6ecc4dc305f30fd69b2ae5549ddc8.jpg)  
Figure 4. Clinical artefact transfer and virtual enhancement. A, Workflow from clinical artefact collection and pattern extraction to cross-domain transfer, strict pairing, restoration and safety review. B, Representative susceptibility, motion, banding, Gibbs-ringing and chemical-shift cases before and after restoration. C, Before-and-after image-quality distributions by artefact type. D, Correlation among distortion, structural and perceptual metrics. E, Comparison with baseline restoration models. F, Artefact-type and severity distribution in the clinical cohort.

## Radiologist assessment separated visual plausibility from clinical usability

A blinded reader study assessed 1,000 held-out cases from four representative tasks, with 250 cases per task (Fig. 5A). Acquired reference, MSCNet and baseline images were de-identified, block-randomized and read in multiple sessions by three radiologists. The five prespecified dimensions were overall image quality, anatomical fidelity, lesion conspicuity, diagnostic confidence and absence of artefact.

MSCNet consistently narrowed the gap to acquired images while maintaining a large separation from the baseline (Fig. 5B–E). Task-level overall image-quality scores for baseline, MSCNet and reference images were 3.28, 4.38 and 4.76 for DWI completion; 3.19, 4.31 and 4.71 for ADC completion; 3.35, 4.42 and 4.74 for T2W completion; and 3.08, 4.17 and 4.63 for T1W completion. Diagnostic-confidence scores showed the same ordering: 3.37, 4.44 and 4.81; 3.26, 4.36 and 4.77; 3.44, 4.49 and 4.79; and 3.15, 4.22 and 4.68. The larger residual gap in T1W completion and lesion conspicuity was consistent with the earlier anatomical analysis rather than hidden by the aggregate score.

Because each task contributed 250 cases, the cross-task arithmetic means were 4.71 for reference images, 4.32 for MSCNet and 3.23 for the strongest baseline for overall image quality, and 4.76, 4.38 and 3.31, respectively, for diagnostic confidence (Extended Data Table 3). Secondary descriptive means were 4.67 versus 4.29 for anatomical fidelity and 4.46 versus 4.07 for lesion conspicuity in lesion-containing cases. Reader- and case-clustered confidence intervals supported the prespecified −0.5-point non-inferiority criterion for overall image quality in DWI (diference −0.38, 95% CI −0.46 to −0.30), ADC (−0.40, −0.48 to −0.32) and T2W completion (−0.32, −0.40 to −0.24), but not T1W completion (−0.46, −0.55 to −0.37). Diagnostic confidence showed the same sequence-specific pattern. Inter-reader agreement for overall image quality was high (ICC 0.87; weighted � = 0.83). Primary and secondary reader analyses are reported in (Supplementary Table 18) and (Supplementary Table 19).

（A）  
![](images/1ea643423be47f17af3b588e5fb448724d76004fb84bbcb3d1608de5a0da8bd1.jpg)

![](images/070e12938ec4d26c600233a8b2961092b1d4d7b1200e478fd973342efd9db168.jpg)

![](images/f201ffc66b661eee66f9e365e4cd00138c82068d917349aee9fec0517bee672c.jpg)

![](images/a62cd2c2abbb7203f33327602c4a7912810a297bf0c4d0d44ef2114f4dba3ec1.jpg)

![](images/dffdb42bacc9cfb0cf56804eb0d3124498b1a5b892087827490cb1425384ea64.jpg)  
（B）

![](images/c21680f6806ec14a9b093f5f8d9751881e6feb99d66bef97b73890a7173b0b2c.jpg)

![](images/d3935ffcd501b40d23b1964d1249d12073993bb094ee7a88212bb41ccb3674f5.jpg)

![](images/9277a9f30cb724f5c1f5c8883856005d4c8952c08aeec43c9dba273cf83c30e6.jpg)

![](images/03fcc7f48f40b5a8d19004c98c3a92a3d87b88d99753d924b00074f2176a40c3.jpg)

![](images/3998d4e8b4b30b5e48f8d40bdfd40bf2e6c6e2ca0edea56539aeab17606ddf88.jpg)

![](images/96c71288fdce69e0a54d903a74cf47d9f7f2f98c4719f3c8354ad4f205fc0312.jpg)

![](images/b4d605a511d120b094725406916f8ac11aceaf8f2ea36d2ad8c595565884862f.jpg)

![](images/8fa41c398156b25c3e1e27d571609645b293baea333891edcb2aca788c5e9489.jpg)

![](images/aa97ec85f86e0a9517b1b47cf0e1c75091d8e1c847b20b20edf23e910426b67d.jpg)

![](images/bd8b2d03bdf06e7eecf4cb5ce81aae266e0535d859f53fd547a6d7f211263cff.jpg)

（D）  
![](images/4bf5ebc20ad8a0c4d8778e7d6739532e63896f612fc7085de928be2ed756461b.jpg)

![](images/4da5f0b7501ad34977f3f411498841d3398f092027df3ab469f59a6f460c3bdf.jpg)

![](images/691b349ada4af496db401e72c3eae8607f8e49876c478b87bd3e0849fa5a31bb.jpg)  
（G）

![](images/6fff4ee595cce804e6a7a55f086d82e688187db56609e27e9620c640441b40be.jpg)

![](images/5a72f7b3edfb3a78f08b64c2f8bed2d424b8107f8122fc6556da12ddfa4b1570.jpg)

![](images/67953cf1ac004bab35069b160bfca12da4699def8d41cd008f6207d11b885d30.jpg)

![](images/c035375c67ab74ab737c52d0aecca7ce63638a9ff5535c00b8ad8c1e0c424668.jpg)

![](images/4195e37c0aad1b7459033c1b9fe7284ac627379a9a3e99a68ceff385cd1812c6.jpg)

![](images/56229ca5206996fdeff085bd1f7a4623670535fd38bd9dccc4ac549cf80b27bf.jpg)  
Figure 5. Blinded radiologist reader study. A, Sampling, task strata, anonymization, block randomization, reader experience, Likert endpoints and statistical plan. B, Mean scores for five clinical dimensions across four completion tasks. C, Overall image-quality distributions. D, Diagnostic-confidence distributions. E, Task-level lesion-conspicuity scores. F, Reference and MSCNet score distributions relative to the prespecified reference-minus-0.5 margin. G, Improvement of MSCNet over the baseline across tasks and clinical dimensions.

## Diagnostic validation and selective prediction defined the usable operating range

A separate 200-case assessment tested whether image completion preserved diagnostic discrimination (Fig. 6A). AUCs for clinically significant prostate cancer were 0.860 with acquired images, 0.841 with MSCNet images and 0.797 with baseline-generated images (Fig. 6B). The MSCNet-minus-reference diference was −0.019 (95% CI, −0.048 to +0.010), compared with −0.063 (−0.094 to −0.032) for the strongest baseline. Sensitivity was 86%, 84% and $7 7 \% ;$ specificity was 82%, 81% and 75%; positive predictive value was 71%, 69% and 61%; and negative predictive value was 92%, 91% and 86%. Calibration remained closer to the acquired condition for MSCNet (ECE 0.038, slope 0.94 and Brier score 0.151) than for the baseline (0.067, 0.82 and 0.184) (Supplementary Table 20). The paired interval is consistent with preservation of diagnostic discrimination under an exploratory −0.05 boundary; because that boundary is not used as a confirmatory prespecified and powered margin here, the result is reported as supportive rather than formal diagnostic non-inferiority.

PI-RADS agreement was evaluated across the 1,000-case reader-study set rather than the 200-case pathology subset. Exact agreement between reference-based and MSCNet-based readings was 77.8%, agreement within one category was 95.6%, and weighted � was 0.871 (Fig. 6D). In the retrospective workflow analysis, median reading time decreased from 5.1 to 3.4 min per case, diagnostic confidence increased from 3.4 to 4.0, and 76% of cases were classified as diagnostic-ready, 17% as usable with caution and 7% as requiring repeat imaging (Fig. 6E,F and Extended Data Table 3). The observed workflow benefit was therefore evaluated as an adjunctive use case in which generated images remain linked to the acquired examination. Complete diagnostic and retrospective workflow endpoints are reported in (Supplementary Table 5).

The locked model was then evaluated in a 186-case external cohort comprising site A, Changhai Hospital, Naval Medical University (72 cases; 39%); site B, Shanghai Second People’s Hospital (64 cases; 34%); and site C, Ningbo Hangzhou Bay Hospital (50 cases; 27%). Site-level SSIM was 0.798, 0.788 and 0.784, respectively, and diagnostic AUC was 0.831, 0.819 and 0.814. The pooled estimates were SSIM 0.791, LPIPS 0.124, MAE 0.061, AUC 0.823 and ECE 0.042 (Supplementary Table 16). A method-by-site interaction was not detected $( P = 0 . 5 5 )$ , although precision was limited by the three external sites (Supplementary Table 24). The native internal risk threshold $( \tau = 0 . 5 0 )$ was transferred without hospital-specific recalibration and retained 37 of 72, 31 of 64 and 23 of 50 cases, respectively; pooled coverage was 48.9% (91 of 186), with failure detection 0.75, false rejection 0.16, retained-case AUC 0.86 and ECE 0.036 (Supplementary Table 25). The three-hospital results support transport of the locked model and threshold while retaining site-level calibration and monitoring as part of deployment.

The remaining question was whether unreliable outputs could be recognized before interpretation. We therefore evaluated the operational reconstruction-risk score generated by the study quality-control pipeline. The score was fitted using development and validation data only; no held-out target or radiologist label was used for test-set adaptation. Regional summaries covered the global image, prostate boundary, peripheral zone and transition zone, whereas lesion-region summaries were used only in annotated retrospective analyses. The score correlated with MAE $( \rho = 0 . 5 8 )$ , LPIPS $( \rho = 0 . 6 1 )$ , SSIM degradation $( \rho = 0 . 4 1 )$ and lesion error $( \rho = 0 . 4 2 )$ (Supplementary Fig. 6). Median scores increased from 0.16 in radiologist-rated reliable cases to 0.43 in cases usable with caution and 0.79 in unreliable cases $( P < 0 . 0 0 1 )$ . Failure-detection AUROC was 0.82 for motion artefact, 0.92 for sequence mismatch, 0.79 for boundary blur and 0.90 for lesion reconstruction error. At the validation-selected threshold of 0.5, failure-detection sensitivity was 0.74, false rejection was 0.18, rejected-case precision was 0.78 and F1 was 0.76. The risk–coverage analysis showed that observed selective risk, defined as one minus diagnostic usability among retained cases, was 0.04 at 25% coverage, 0.08 at 50% coverage, 0.14 at $7 5 \%$ coverage and 0.21 without gating. Thus, the balanced policy achieved diagnostic usability of 0.92 while retaining 50% of outputs and was not interpreted as reliability at full clinical coverage. The full internal risk–coverage curve and operating points are reported in (Supplementary Fig. 6) and (Supplementary Table 8), and the fixed-threshold external transport results in (Supplementary Table 25). No hospital-specific recalibration, prospective decision-curve analysis or clinical net-benefit analysis was

![](images/c770b6ab2a818c1e1d2ca5e369cd2771a0527d715bb9407412be73e52373c0aa.jpg)

## performed.

(A)  
![](images/a1c62a8f4e6170758b899839c36198eccd6679d245182a8477c4cc1d5ae34e53.jpg)  
Stage 1: Case pool

![](images/7a83f5eb8c4b62e1c7a83035e8b55257d0440b5fd85d571e57dc8c82afefa35b.jpg)

![](images/ff2760b0ff38a707cce76581ee1b82dc4221f403630516045b81f706df38012f.jpg)  
Stage 2: Image conditions

![](images/9df0617239d2610864fbaa18b53c4ba6291c5d26dbe5882f6d162328e974a7c5.jpg)  
Stage 3:

![](images/ca41f414d16029b0a97ab9679720d281fc71de3776a8a4dac146a190608a8a47.jpg)

![](images/35250496f1524467f87ef769f29f6623aa67fe4f258252981e60054e825d8c94.jpg)

![](images/5d3699cd684b6bdd25781d1aaf6d081d6448d29567b2cd23f25bf865f8a1ea2b.jpg)  
Randomized and blinded  
Stage 4: Radiologist assessment  
Source labels hidden

![](images/f6ba7043f7430727fe6f93e14b907a57bbae7a15b6dd4f8104ef202f2ab1f3b3.jpg)  
Pathology/Clinical follow-up Gold standard&Endpoints

![](images/7fd2dae5f24011543c1a33aeeceec7fdbd0b81027fc58e68a31cb7acbf5e9e4d.jpg)

![](images/a2c2492b01565e5560f36f7a59f34b100fa1b4346ce123ea3e157a58c53b2b55.jpg)

![](images/18167b3228f31dd73c900739998004bf853846b7601432a0ef671cb259585c78.jpg)

![](images/7964a3b6398001080dc9178b4f2daec3d7f88bb2954d350470bc76651d71ddcd.jpg)

![](images/e492cf65eef0173c7005eef0baa2a1f9bfe908153ba930897b73f03109b3098a.jpg)  
Source labels hidden

![](images/d80db0edb60b39252918f93b530b7441aff24056de2e717b9c9b552a0d1e3f34.jpg)  
csPCa performance

![](images/8bdde930a627eb788858a0dfb9be5b91a2336eea3a5a655f8118e3bb0010869c.jpg)

![](images/a50077983ebf802cf086973a7d82f00361041b2d2fceda277e5cadef8341823c.jpg)  
Diagnostic validation cohort N = 200 Biopsy and follow-up available  
Reader washout period (2-4weeks)  
AUC: 0.841 Sensitivity: 84% Specificity: 81%

![](images/741360dc9b94fa0f31d8d2e93b0d1784bd06627a64be20b9727321b5bd21ec3e.jpg)  
Lesion localization was evaluated separately in panel C. Values correspond to the MSCNet reading condition.

(B)  
![](images/c212ce9c3d62d5fdd629e2417f828ea8bf7b85d11a64312398cc66853bab1cb5.jpg)

![](images/9cd296817d1a38d537fdbc354975a0168ec9c3859b10263e0f5d45260364696a.jpg)

(C)  
![](images/93ddfe20082e53a74193fb2c8bded81ea5cc216fc39f26c5dcabf09fe8fb1068.jpg)

![](images/201f578fb9203693206664107e4d7652c902c953eb78f72f3e0f16e22809561f.jpg)

![](images/9db475852d746f2aaea5250f469b2dd472ea58b6607ef4b20b038328aa6d8552.jpg)

![](images/4c8ec15f33ab4db52c4f498bc5a32b03b1b39728f76d1526c4a885d0d474f7c0.jpg)  
Image condition ■ GT repeat □ MSCNet Baseline

(D)  
![](images/794683ed7fb5cd7980d760450d217903e54c78a6536e24949b05cd7fbe2bfc54.jpg)  
(E)

![](images/eff9ff29b257726cd0ca7aca44b889fdb1a1ecc43610406f41c1d6ca0b03a832.jpg)

![](images/2d5faab44dd7ee7e5859887e496fe8bd560f755de3645ad2e23afb933104ef66.jpg)

![](images/7927a0131a794d4fed392b228c179fb62a83bc54be2128a6990547fdc9b156e8.jpg)  
(F)  
Al-based reading (MSCNet)

![](images/e47bdd53e03439dbbaa86f860bca0c907f2ac28dc0b8a27ba488f9f1431fad2f.jpg)

![](images/1684b8b241d923b272d332c030581a12b96d5007fe3a574ffc6ecf82686fab5f.jpg)

![](images/5c66b981fef864e720e4a8d63b4971e57186a4f346274cd3608455e6c2b44c89.jpg)  
Figure 6. Diagnostic validation and clinical usability. A, Separate diagnostic cohort, image conditions, blinded reading and pathology or clinical follow-up reference standard. Summary values in the endpoint panel correspond to the MSCNet condition and match the quantitative results in B. B, AUC, sensitivity, specificity, positive predictive value, negative predictive value and AUC diference for acquired, MSCNet and baseline images. C, Lesion Dice, centre-distance error, volume overlap and boundary-distance error. D, PI-RADS agreement matrix in the 1,000-case reader-study set. E, Paired reading time and diagnostic confidence before and after AI assistance. F, Retrospective usability categories after AI assistance.

## Discussion

This study treats incomplete prostate mpMRI as a problem of information routing, clinical fidelity and selective use rather than image translation alone. The principal task-specific MSCNet models retained the highest SSIM after the comparison set was expanded to unified synthesis, matched-capacity Transformers, modern difusion models, loss-matched convolutional controls, retrieval and physics-aware baselines. The smaller but persistent advantage under these stronger controls is more informative than the larger gap to legacy baselines: it shows that capacity and optimization explain part, but not all, of the result.

The task hierarchy also constrains the mechanistic and clinical claims. DWI and ADC share difusionderived information and should not be treated as equivalent evidence to reconstruction of T2W anatomy. Separating tasks showed mean SSIM advantages of 0.018 in the physically coupled group, 0.024 in crosscontrast T2W synthesis and 0.020 for T1W completion. No universal change in SSIM can be assumed to represent a clinically meaningful diference, and the 0.020 ten-task gain is therefore interpreted as a technical endpoint rather than a diagnostic surrogate. Its clinical-anatomical relevance is supported by the matched-control subset, in which the advantage was larger for lesion fidelity (+0.05) and boundary SSIM (+0.055), together with independent reader, safety and diagnostic analyses. This triangulation supports the interaction between sequence-specific routing, global anatomical context and boundary supervision rather than a claim based on parameter count or global SSIM alone.

The shared-model experiments define both an opportunity and a boundary. One 158.3-million-parameter model reproduced all ten tasks with a mean SSIM reduction of 0.015 relative to the optimized task-specific models, and unified pretraining followed by task-specific fine-tuning recovered the optimized task-specific mean. Zero-shot unseen-combination performance remained lower, however. The evidence therefore supports a clinically evaluated task-specific framework with a unified routing extension; zero-shot performance remained outside the principal claim.

Clinical validation progressed from controlled to less curated settings. Clinically derived artefact transfer isolated five corruption classes, while 204 same-patient repeat acquisitions tested genuine degraded scans against a clean repeat reference. The repeat-scan improvement from SSIM 0.714 to 0.883 supports realacquisition recovery, but repeated examinations can difer in positioning, physiology and interval change and remain a retrospective reference rather than a perfect ground truth. Similarly, the pooled 186-case external cohort (site A, 72 cases; site B, 64; site C, 50) showed only a modest completion shift and retained an AUC of 0.823. Changhai Hospital and Shanghai Second People’s Hospital were institutionally independent of the development centre, whereas Ningbo Hangzhou Bay Hospital is a geographically distinct hospital afiliated with Ren Ji Hospital; the analysis therefore supports multicentre transportability but not validation across three fully independent health systems.

Reader and diagnostic analyses exposed sequence-specific limits that aggregate means would hide. Overall image-quality non-inferiority was supported for DWI, ADC and T2W, but not T1W, and diagnostic confidence showed the same pattern. The 200-case AUC diference between MSCNet and acquired images was small, with a confidence interval that remained above a supportive −0.05 boundary; formal diagnostic non-inferiority nevertheless depends on whether that boundary and its power analysis were prespecified before unblinding. These results support preservation of diagnostic information within the evaluated workflow while keeping the acquired examination as the clinical reference.

Selective prediction provides a practical response to the remaining failures, but reliability cannot be separated from coverage. Internally, diagnostic usability increased from 0.79 at full coverage to 0.92 at the validation-selected 50% operating point. When the native threshold � = 0.50 was transferred externally, it retained 48.9% of cases, with failure detection 0.75, false rejection 0.16, retained-case AUC 0.86 and ECE 0.036. This site-level degradation shows why threshold transport must be reported explicitly and monitored by hospital. Generated images should remain labelled, displayed beside acquired sequences and withheld when predicted risk is high, anatomy is incomplete or the inputs are severely corrupted.

Limitations remain. The pooled external cohort combined three hospitals, and hospital-specific acquisition distributions, patient flow and calibration may influence the aggregate estimate. Repeat-scan analyses require patient-clustered inference and careful accounting for treatment or interval change. Generated ADC maps should not be used for quantitative ADC measurement without separate calibration, missing anatomical coverage cannot be recreated from absent information, and T1W completion does not recover dynamic contrast kinetics. Prospective multicentre workflow studies, locked health-economic endpoints and post-deployment calibration monitoring remain necessary. Within these boundaries, MSCNet supports a reliability-gated approach to cross-modal completion and enhancement of prostate MRI.

## Online Methods

## Cohorts, governance and task definition

The study used public and hospital-based prostate MRI data. PI-CAI contained T2W, ADC and high-b-value DWI; the vendor-specific PROSTATEx analysis contained T2W, DWI and ADC.<sup>9;10</sup> The institutional threesequence and four-sequence cohorts, clinical artefact cohort, reader study and internal diagnostic cohort were obtained at Ren Ji Hospital, Shanghai Jiao Tong University School of Medicine, Shanghai, China. The clinical artefact cohort contained 182 unique patients and 262,306 DICOM slices. A locked external cohort contained 186 cases: site A, 72 cases (39%) from Changhai Hospital, Naval Medical University, Shanghai, China; site B, 64 cases (34%) from Shanghai Second People’s Hospital, Shanghai, China; and site C, 50 cases (27%) from Ningbo Hangzhou Bay Hospital, Ningbo, Zhejiang, China. All external cases had paired completion targets and pathology-confirmed outcomes.

All development splits were performed at the patient level. PI-CAI was divided into 1,000 development, 200 validation and 276 held-out cases; the institutional three-sequence cohort into 2,232, 318 and 639; and the four-sequence cohort into 1,111, 157 and 327. The external cohort, reader study, diagnostic cohort and repeat-scan analyses were not used to select principal model weights. Hospital data were processed retrospectively after de-identification under institutional ethics approval and data-governance agreements. The requirement for written informed consent was waived for retrospective secondary analysis under the applicable local procedures. The verified Ren Ji approval identifier and hospital-level governance roles are reported in the Ethics statement and (Supplementary Table 26).

For three-sequence data, the targets were DWI from T2W and ADC; ADC from T2W and DWI; and T2W from ADC and DWI. For four-sequence data, the targets were T1W from T2W, DWI and ADC; ADC from T2W, DWI and T1W; DWI from T2W, ADC and T1W; and T2W from DWI, ADC and T1W. The principal model was trained independently for each target configuration. A secondary MSCNet-Shared experiment used one model for all targets, with an available-modality mask, target-modality token and training-time modality dropout. Cohort governance, public-dataset overlap, shared-model data use and exclusion rules are detailed in (Supplementary Note 1) and (Supplementary Note 10).

## Preprocessing, matching and quality control

DICOM series were identified from metadata and verified using rule-based and manual checks. Images were standardized for orientation, resampled to a common analysis grid and aligned within each examination. Sequence-specific intensity normalization was applied after spatial processing. The prostate region and adjacent context were retained for model input. Cases with corrupted files, incomplete metadata, severe non-correctable misregistration, unreliable lesion annotation or non-diagnostic image content were excluded.

Sequence correspondence was checked at the patient, examination, slice and anatomical-region levels. Rigid or afine alignment was followed by deformable correction when required. Quality was measured using landmark displacement, normalized mutual information, prostate-region Dice and centre displacement. The correction and exclusion analyses are shown in (Supplementary Fig. 4); complete rules are provided in (Supplementary Note 2).

## MSCNet

At scale �, modality-specific encoders produced feature maps $F _ { m } ^ { ( s ) }$ for each available modality �. Each encoder used residual blocks<sup>17</sup> with squeeze-and-excitation recalibration<sup>23</sup> at five scales with 32, 64, 128,

256 and 512 channels. The scale-specific feature tensor was

$$
\boldsymbol { C } ^ { ( s ) } = \operatorname { C o n c a t } \left( \boldsymbol { F } _ { 1 } ^ { ( s ) } , \ldots , \boldsymbol { F } _ { M } ^ { ( s ) } \right) .
$$

A gating network generated modality- and location-dependent weights

$$
g _ { m } ^ { ( s ) } = \sigma \Bigl ( \phi _ { m } ^ { ( s ) } ( C ^ { ( s ) } ) \Bigr ) ,
$$

and the fused representation was

$$
\widetilde { F } ^ { ( s ) } = \psi ^ { ( s ) } \Big ( \mathrm { C o n c a t } \Big ( g _ { 1 } ^ { ( s ) } \odot F _ { 1 } ^ { ( s ) } , . . . , g _ { M } ^ { ( s ) } \odot F _ { M } ^ { ( s ) } \Big ) \Big ) .
$$

Here, $\sigma$ denotes the sigmoid function, ⊙ element-wise multiplication and $\psi ^ { ( s ) }$ the merge transform. This formulation allowed the contribution of each sequence to vary by feature scale and spatial location.

The deepest fused representation was tokenized and processed by a multi-head self-attention bottleneck.<sup>21</sup> Decoder features were upsampled and combined with fused encoder features through attention gates.<sup>22</sup> Auxiliary predictions at intermediate scales provided deep supervision.<sup>24</sup> An edge branch compared gradients of the prediction and acquired target. Further implementation details are given in (Supplementary Note 3).

For MSCNet-Shared, absent sequences were represented by zero-filled tensors accompanied by a binary availability mask, and a learned target token specified the requested output. Training sampled valid available-target combinations and applied modality dropout without exposing held-out test cases. The shared model used 158.3 million parameters. A unified pretraining analysis used 3,750 development cases (1,200 PI-CAI development/validation cases and 2,550 institutional three-sequence development/validation cases), followed by task-specific fine-tuning, few-shot adaptation or frozen-encoder transfer. These experiments were secondary and did not replace the task-specific principal models ((Supplementary Table 12) and (Supplementary Table 13)).

## Optimization and comparison models

The full objective was

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { c h a r } } + 0 . 1 5 \mathcal { L } _ { \mathrm { S S I M } } + 0 . 1 0 \mathcal { L } _ { \mathrm { M S } , \mathrm { S S I M } } + 0 . 0 2 \mathcal { L } _ { \mathrm { p e r c } } + 0 . 0 5 \mathcal { L } _ { \mathrm { c d g c } } + 0 . 1 0 \mathcal { L } _ { \mathrm { w a v e } } + 0 . 0 5 \mathcal { L } _ { \mathrm { f r e q } } + \lambda _ { \mathrm { d s } } \mathcal { L } _ { \mathrm { d c p } } . } \end{array}
$$

The reconstruction terms follow established image-restoration objectives,<sup>25–27;45;46</sup> and $\lambda _ { \mathrm { d s } }$ denotes the normalized contribution of the auxiliary decoder outputs. Task-specific MSCNet contained 154.1 million parameters and was trained for 200 epochs with AdamW,<sup>44</sup> an initial learning rate of $2 \times 1 0 ^ { - 4 }$ , batch size 2, cosine annealing and 10 warm-up epochs.

The legacy comparison set contained DynUNet implemented in MONAI,<sup>20</sup> Pix2Pix,<sup>19</sup> a residual generator,<sup>17</sup> an LSGAN model<sup>39</sup> with a PatchGAN discriminator and spectral normalization,<sup>40</sup> and a conditional difusion model based on 1,000 difusion steps and 50-step DDIM sampling .<sup>41–43</sup> The expanded comparison set added a unified missing-modality model, a multi-contrast Transformer, modality-masked difusion, a structure-aware latent-difusion model, a parameter-matched Transformer, wide DynUNet, DynUNet trained with the MSCNet objective, conditional flow matching, nearest-neighbour retrieval and target-specific signal-model or protocol-regression controls. Data splits, target definitions and evaluation masks were identical. Hyperparameters were selected on validation data with a matched search budget. Complete implementation and fairness details are reported in (Supplementary Table 9)–(Supplementary Table 11) and (Supplementary Note 10).

## Image, spatial and lesion-level evaluation

Image fidelity was measured using PSNR, SSIM,<sup>25</sup> MS-SSIM,<sup>26</sup> LPIPS,<sup>27</sup> FID<sup>28</sup> and MAE. Spatial evaluation partitioned error into gland, boundary-ring and peri-gland regions and stratified results by apex, mid-gland and base. Volumetric consistency included adjacent-slice similarity, slice-to-slice variation, volumetric smoothness and z-trend correlation. Boundary preservation included edge preservation, boundary SSIM, gradient similarity and capsule sharpness.

Lesion fidelity was assessed using lesion-to-background contrast-to-noise ratio, lesion-region signal-tonoise ratio, contrast ratio, line-profile agreement and radiologist conspicuity. Lesion localization used Dice overlap,<sup>29</sup> centre distance, volume overlap and boundary distance. Evaluation definitions and aggregation rules are given in (Supplementary Note 5).

## Artefact transfer and same-patient repeat-scan evaluation

Clinically observed artefact patterns were extracted from low-quality acquisitions and categorized as susceptibility, motion ghosting, banding, Gibbs ringing or chemical shift. Candidate components were screened for topology and texture, aligned with clean targets and transferred in image or frequency space. Pairing required anatomical correspondence, lesion-boundary preservation and SSIM of approximately 0.8 between the clean target and controlled degraded image. Restoration was evaluated against the quality controlled target with six image-quality metrics and radiologist safety review.

A distinct repeat-scan analysis used 204 degraded–repeat pairs from 182 unique patients. One patient could contribute more than one adjudicated stratum. The clean reference was a same-patient repeat acquisition obtained within four weeks; treatment status, interval, scanner and protocol correspondence were recorded. Categories were motion blur, susceptibility distortion, low-SNR high-b-value DWI, geometric distortion, zipper artefact, ghosting and radiofrequency inhomogeneity. Statistical inference used patient-clustered paired comparisons. The controlled-transfer and repeat-scan taxonomies were analysed separately ((Supplementary Table 3), (Supplementary Table 17) and (Supplementary Note 11)).

## Reader study, diagnostic assessment and external evaluation

The reader study sampled 1,000 held-out cases across four tasks, with 250 cases per task. Acquired, MSCNet and baseline images were de-identified and block-randomized. Three radiologists read the images in sessions with washout and scored overall quality, anatomical fidelity, lesion conspicuity, diagnostic confidence and artefact absence on five-point scales. The acquired-reference comparison used a prespecified −0.5-point margin. Primary task-level diferences and 95% confidence intervals were estimated with models retaining reader and case clustering; non-inferiority was supported only when the lower confidence bound exceeded the margin. Agreement was summarized by ICC and weighted �.<sup>31;32</sup>

The diagnostic assessment used 200 cases with biopsy or clinical follow-up. Readers assigned PI-RADS version 2.1 categories and binary clinically significant prostate cancer judgements. AUCs were compared using paired methods,<sup>30</sup> and calibration was summarized by expected calibration error, calibration slope and Brier score. The −0.05 AUC boundary was used only as an exploratory supportive benchmark and was not treated as a confirmatory non-inferiority margin. PI-RADS agreement was calculated separately in the 1,000-case reader cohort.

The external cohort contained 186 cases across Changhai Hospital, Naval Medical University (site A; 72 cases; 39%), Shanghai Second People’s Hospital (site B; 64 cases; 34%) and Ningbo Hangzhou Bay Hospital (site C; 50 cases; 27%), with paired completion targets and pathology-confirmed outcomes. Site A used a 3.0-T Siemens MAGNETOM Prisma system, site B a 3.0-T Philips Ingenia system and site C a 1.5-T GE SIGNA Artist system; protocol diferences included T2W resolution, slice thickness and DWI �-values (Supplementary Table 23). The trained reconstruction models, preprocessing rules and internal risk threshold were fixed before external evaluation. Completion and diagnostic endpoints were reported separately even when the same case contributed to both. No hospital-specific training, fine-tuning or threshold recalibration was performed. Site-level case spectrum, performance, interaction and threshold-transport results are provided in (Supplementary Table 16) and (Supplementary Table 22)–(Supplementary Table 25).

## Safety adjudication, uncertainty and rejection

Safety events were defined as suspected false lesion, reduced lesion conspicuity, blurred lesion margin, partial lesion erasure, non-diagnostic output or input-output inconsistency. Events were adjudicated by the radiologists and assigned to acquisition-dominant, lesion-dominant, registration/model-dominant, mixed or unassigned contributors.

The quality-control analysis used a normalized voxel-wise reconstruction-risk map, denoted $U ( x )$ , generated by the implementation archived with the review code. Model fitting and operating-point selection used development and validation data only, and the risk score was fixed before evaluation on held-out internal and external cohorts. The score was interpreted operationally as reconstruction risk rather than as a decomposition of aleatoric and epistemic uncertainty.

The primary case-level score was the maximum of the 95th-percentile values within the global image, prostate boundary, peripheral zone and transition zone. Lesion-region summaries were calculated only retrospectively in cases with lesion annotations and were not required for the primary score. Failure detection was evaluated by AUROC for motion artefact, sequence mismatch, boundary blur and lesion reconstruction error. Thresholds from 0.2 to 0.7 were assessed using failure-detection sensitivity, false-rejection rate, rejected-case precision and F1. Coverage was the proportion of outputs retained below a threshold. Selective risk was the mean failure indicator among retained cases; for the clinical-usability analysis it was additionally reported as one minus diagnostic usability. Risk–coverage curves were generated by sweeping the fixed score threshold. Expected calibration error was computed from fixed-width score bins. The archived code and configuration files specify the estimator implementation used for the reported experiments.

The threshold of 0.5 and the conservative, balanced and permissive policies were selected on validation data and evaluated internally without test-set retuning. The locked internal threshold was then transferred directly to the 186-case external cohort without hospital-specific recalibration. Coverage, retained count, failure detection, false rejection, retained-case AUC and calibration were reported jointly for external threshold transport. No decision-curve, cost-efectiveness or prospective clinical net-benefit analysis was performed. Exact definitions and external transport results are provided in (Supplementary Note 9), (Supplementary Note 13), (Supplementary Table 8), (Supplementary Table 21) and (Supplementary Table 25).

## Statistical analysis and reporting

Continuous values were summarized by mean and standard error or median and interquartile range, as appropriate. Paired normally distributed diferences used paired �-tests; non-normal paired or ordinal outcomes used Wilcoxon signed-rank tests. Patient clustering was retained for multiple repeat acquisitions. Correlations were summarized using Spearman’s $\rho .$ . AUC comparisons used paired DeLong or paired bootstrap procedures.<sup>30</sup> Weighted $\kappa ^ { 3 1 }$ and $\mathrm { I C C } ^ { 3 2 }$ quantified reader agreement. Reader non-inferiority models retained crossed reader and case efects, and conclusions were based on confidence-interval lower bounds rather than point estimates alone. Multiple comparisons were controlled using the Benjamini–Hochberg procedure.<sup>33</sup>

Thresholds were chosen on validation data; held-out internal and external risk–coverage results were reported with corresponding coverage and without test-set retuning. External threshold transport was assessed using the native internal threshold. All model and task summaries were calculated at case level, and task groups were prespecified as physically coupled DWI–ADC, cross-contrast T2W and auxiliary T1W. No formal health-economic, cost-efectiveness or decision-curve analysis was conducted. Reporting was guided by CLAIM, STARD and STARD-AI.<sup>34–36</sup>

## Ethics approval and consent

The retrospective institutional study was approved by the Ethics Committee of Ren Ji Hospital, Shanghai Jiao Tong University School of Medicine (approval no. KY2018-212-s2) and was conducted in accordance with the Declaration of Helsinki and applicable data-protection requirements. The requirement for written informed consent for retrospective use of de-identified institutional data was waived by the approving committee. De-identified external data from Changhai Hospital, Naval Medical University, Shanghai Second People’s Hospital and Ningbo Hangzhou Bay Hospital were analysed under the applicable local ethics, consent-waiver and data-governance procedures and institutional data-use agreements. Public PI-CAI and PROSTATEx data were used under their respective access and governance conditions.

## Data availability

PI-CAI and PROSTATEx are available from their public repositories subject to the applicable access terms. Institutional imaging and clinical data are not publicly released because they contain protected health information. Requests for access to de-identified institutional data should be directed to the co-corresponding authors and will be reviewed by the relevant institutional data-access committee. Access requires an approved research purpose, applicable ethics approval, a data-use agreement and compliance with relevant privacy and data-protection requirements.

## Code availability

Code, configuration files and inference instructions for the reported analyses are available to editors and reviewers at https://pcnkfl2i3xls.feishu.cn/file/EQysbKKYDoWgZdxhhbMcbjsqnib. A versioned public repository and archival DOI are planned for the published record.

## Funding

This work was supported by the National Natural Science Foundation of China (No. 82371912) and the Science and Technology Commission of Shanghai Municipality (No. YDZX20243100003001).

## Acknowledgements

The authors thank the participating radiologists and the clinical data-management teams at Ren Ji Hospital, Changhai Hospital, Shanghai Second People’s Hospital and Ningbo Hangzhou Bay Hospital for support with blinded review, data curation and governance.

## Author contributions

All authors made substantial contributions to one or more aspects of study conception, methodology, investigation, data curation, formal analysis, software, validation, visualization or manuscript preparation. All authors reviewed and approved the final manuscript. Yang Liu, Dacheng Tao and Guangyu Wu are co-corresponding authors.

## Competing interests

The authors declare no competing interests.

## References

[1] Turkbey, B. et al. Prostate imaging reporting and data system version 2.1: 2019 update of prostate imaging reporting and data system version 2. European urology 76, 340–351 (2019).

[2] Ahmed, H. U. et al. Diagnostic accuracy of multi-parametric mri and trus biopsy in prostate cancer (promis): a paired validating confirmatory study. The Lancet 389, 815–822 (2017).

[3] Kasivisvanathan, V. et al. Mri-targeted or standard biopsy for prostate-cancer diagnosis. New England Journal of Medicine 378, 1767–1777 (2018).

[4] Drost, F.-J. H. et al. Prostate magnetic resonance imaging, with or without magnetic resonance imaging-targeted biopsy, and systematic biopsy for detecting prostate cancer: a cochrane systematic review and meta-analysis. European urology 77, 78–94 (2020).

[5] Tan, C. H., Wei, W., Johnson, V. & Kundra, V. Difusion-weighted mri in the detection of prostate cancer: meta-analysis. American Journal of Roentgenology 199, 822–829 (2012).

[6] Mazaheri, Y., Vargas, H. A., Nyman, G., Akin, O. & Hricak, H. Image artifacts on prostate difusionweighted magnetic resonance imaging: trade-ofs at 1.5 tesla and 3.0 tesla. Academic radiology 20, 1041–1047 (2013).

[7] Giganti, F. et al. Prostate imaging quality (pi-qual): a new quality control scoring system for multiparametric magnetic resonance imaging of the prostate from the precision trial. European urology oncology 3, 615–619 (2020).

[8] de Rooij, M. et al. Pi-qual version 2: an update of a standardised scoring system for the assessment of image quality of prostate mri. European radiology 34, 7068–7079 (2024).

[9] Saha, A. et al. Artificial intelligence and radiologists in prostate cancer detection on mri (pi-cai): an international, paired, non-inferiority, confirmatory study. The Lancet Oncology 25, 879–887 (2024).

[10] Armato III, S. G. et al. Prostatex challenges for computerized classification of prostate lesions from multiparametric magnetic resonance images. Journal ofMedical Imaging 5, 044501–044501 (2018).

[11] Chartsias, A., Joyce, T., Giufrida, M. V. & Tsaftaris, S. A. Multimodal mr synthesis via modalityinvariant latent representation. IEEE transactions on medical imaging 37, 803–814 (2017).

[12] Havaei, M., Guizard, N., Chapados, N. & Bengio, Y. Hemis: Hetero-modal image segmentation. In International conference on medical image computing and computer-assisted intervention, 469–477 (Springer, 2016).

[13] Zhou, T., Fu, H., Chen, G., Shen, J. & Shao, L. Hi-net: hybrid-fusion network for multi-modal mr image synthesis. IEEE transactions on medical imaging 39, 2772–2781 (2020).

[14] Sharma, A. & Hamarneh, G. Missing mri pulse sequence synthesis using multi-modal generative adversarial network. IEEE transactions on medical imaging 39, 1170–1183 (2019).

[15] Zhang, Y. et al. Unified multi-modal image synthesis for missing modality imputation. IEEE Transactions on Medical Imaging 44, 4–18 (2024).

[16] Ronneberger, O., Fischer, P. & Brox, T. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, 234–241 (Springer, 2015).

[17] Shafiq, M. & Gu, Z. Deep residual learning for image recognition: A survey. Applied sciences 12, 8972 (2022).

[18] Goodfellow, I. J. et al. Generative adversarial nets. Advances in neural information processing systems 27 (2014).

[19] Isola, P., Zhu, J.-Y., Zhou, T. & Efros, A. A. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, 1125–1134 (2017).

[20] Cardoso, M. J. et al. Monai: An open-source framework for deep learning in healthcare. arXiv preprint arXiv:2211.02701 (2022).

[21] Vaswani, A. et al. Attention is all you need. Advances in neural information processing systems 30 (2017).

[22] Oktay, O. et al. Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999 (2018).

[23] Hu, J., Shen, L. & Sun, G. Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, 7132–7141 (2018).

[24] Lee, C.-Y., Xie, S., Gallagher, P., Zhang, Z. & Tu, Z. Deeply-supervised nets. In Artificial intelligence and statistics, 562–570 (Pmlr, 2015).

[25] Wang, Z., Bovik, A. C., Sheikh, H. R. & Simoncelli, E. P. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13, 600–612 (2004).

[26] Wang, Z., Simoncelli, E. P. & Bovik, A. C. Multiscale structural similarity for image quality assessment. In The thrity-seventh asilomar conference on signals, systems & computers, 2003, vol. 2, 1398–1402 (Ieee, 2003).

[27] Zhang, R., Isola, P., Efros, A. A., Shechtman, E. & Wang, O. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, 586–595 (2018).

[28] Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B. & Hochreiter, S. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017).

[29] Dice, L. R. Measures of the amount of ecologic association between species. Ecology 26, 297–302 (1945).

[30] DeLong, E. R., DeLong, D. M. & Clarke-Pearson, D. L. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics 837–845 (1988).

[31] Cohen, J. Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit. Psychological bulletin 70, 213 (1968).

[32] Koo, T. K. & Li, M. Y. A guideline of selecting and reporting intraclass correlation coeficients for reliability research. Journal of chiropractic medicine 15, 155–163 (2016).

[33] Benjamini, Y. & Hochberg, Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal ofthe Royal statistical society: series B (Methodological) 57, 289–300 (1995).

[34] Mongan, J., Moy, L. & Kahn Jr, C. E. Checklist for artificial intelligence in medical imaging (claim): a guide for authors and reviewers (2020).

[35] Bossuyt, P. M. et al. Stard 2015: an updated list of essential items for reporting diagnostic accuracy studies. Radiology 277, 826–832 (2015).

[36] Sounderajah, V. et al. The stard-ai reporting guideline for diagnostic accuracy studies using artificial intelligence. Nature medicine 31, 3283–3289 (2025).

[37] Zhu, J.-Y., Park, T., Isola, P. & Efros, A. A. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision, 2223–2232 (2017).

[38] Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J. & Maier-Hein, K. H. nnu-net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods 18, 203–211 (2021).

[39] Mao, X. et al. Least squares generative adversarial networks. In Proceedings of the IEEE international conference on computer vision, 2794–2802 (2017).

[40] Miyato, T., Kataoka, T., Koyama, M. & Yoshida, Y. Spectral normalization for generative adversarial networks. arXiv preprint arXiv:1802.05957 (2018).

[41] Ho, J., Jain, A. & Abbeel, P. Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020).

[42] Song, J., Meng, C. & Ermon, S. Denoising difusion implicit models. arXiv preprint arXiv:2010.02502 (2020).

[43] Nichol, A. Q. & Dhariwal, P. Improved denoising difusion probabilistic models. In International conference on machine learning, 8162–8171 (PMLR, 2021).

[44] Loshchilov, I. & Hutter, F. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017).

[45] Jiang, L., Dai, B., Wu, W. & Loy, C. C. Focal frequency loss for image reconstruction and synthesis. In Proceedings of the IEEE/CVF international conference on computer vision, 13919–13929 (2021).

[46] Zhao, H., Gallo, O., Frosio, I. & Kautz, J. Loss functions for image restoration with neural networks. IEEE Transactions on computational imaging 3, 47–57 (2016).

## Extended Data

Extended Data Table 1. Study cohorts and analytical roles. Splits were defined at the patient level. PROSTATEx was treated as a vendor-specific public cohort rather than independent external validation.
<table><tr><td>Cohort</td><td>Source</td><td>Sequences</td><td>Size or split</td><td>Primary role</td></tr><tr><td>PI-CAI</td><td>Public, multicentre</td><td>T2W, ADC, high-b-value DWI</td><td>1,476; 1,000/200/276</td><td>Public-domain development and held-out evaluation of</td></tr><tr><td>PROSTATEx</td><td>Public, Siemens</td><td>T2W, DWI, ADC</td><td>346</td><td>2-to-1 tasks Vendor-specific public-cohort analysis</td></tr><tr><td>Institutional three-sequence</td><td>Ren Ji Hospital, Shanghai, China</td><td>T2W, DWI, ADC</td><td>3,189; 2,232/318/639</td><td>Training, validation and testing of 2-to-1</td></tr><tr><td>Institutional four-sequence</td><td>Ren Ji Hospital, Shanghai, China</td><td>T2W, DWI, ADC, T1W</td><td>1,595; 1,111/157/327</td><td>tasks Training, validation and testing of 3-to-1 tasks</td></tr><tr><td>Clinical artefact</td><td>Ren Ji Hospital archive</td><td>Degraded prostate MRI sequences</td><td>182 patients; 262,306 DICOM slices; 204 repeat</td><td>Controlled artefact transfer, same-patient repeat-scan validation and safety evaluation</td></tr><tr><td>Reader study</td><td>Ren Ji Hospital held-out cohort</td><td>Four representative tasks</td><td>pairs 1,000; 250 per task</td><td>Three-reader blinded assessment csPCa discrimination</td></tr><tr><td>Diagnostic assessment</td><td>Ren Ji Hospital separate clinical set</td><td>Reference, MSCNet and baseline conditions</td><td>200</td><td>and workflow endpoints Locked pooled</td></tr><tr><td>Three-hospital external cohort</td><td>Site A: Changhai Hospital (72; 39%); site B: Shanghai Second People&#x27;s Hospital (64; 34%); site C: Ningbo Hangzhou Bay Hospital (50; 27%)</td><td>T2W, DWI, ADC</td><td>186 total</td><td>completion, diagnostic and threshold-transport evaluation</td></tr></table>

Extended Data Table 2. Main quantitative performance of MSCNet. Values in parentheses are absolute changes relative to the strongest non-MSCNet baseline. PSNR and SSIM are higher-is-better; LPIPS and MAE are lower-is-better.
<table><tr><td>Dataset</td><td>Task</td><td>Input → target</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>MAE</td></tr><tr><td>PI-CAI</td><td>P1</td><td> $\mathrm { T 2 W + A D C  h i g h – b \ D W I }$ </td><td>25.91 (+0.53)</td><td>0.8600 (+0.0180)</td><td>0.0890 (−0.0100)</td><td>0.0384 (-0.0036)</td></tr><tr><td>PI-CAI</td><td>P2</td><td> $\mathrm { T 2 W + h i g h – b ~ D W I  A D C }$ </td><td>28.43 (+0.48)</td><td>0.8890 (+0.0170)</td><td>0.0700 (−0.0110)</td><td>0.0261 (-0.0029)</td></tr><tr><td>PI-CAI</td><td>P3</td><td> $\mathrm { A D C } + \mathrm { h i g h – b } \ \mathrm { D W I }  \mathrm { T 2 W }$ </td><td>23.07 (+0.66)</td><td>0.7300 (+0.0240)</td><td>0.1480 (-0.0160)</td><td>0.0573 (-0.0047)</td></tr><tr><td>Institutional</td><td>A</td><td> $\mathrm { T 2 W + A D C }  \mathrm { D W I }$ </td><td>24.17 (+0.46)</td><td>0.8460 (+0.0180)</td><td>0.0950 (-0.0110)</td><td>0.0621 (−0.0039)</td></tr><tr><td>Institutional</td><td>B</td><td>T2W + DWI → ADC</td><td>27.46 (+0.44)</td><td>0.8660 (+0.0180)</td><td>0.0830 (-0.0100)</td><td>0.0382 (−0.0038)</td></tr><tr><td>Institutional</td><td>C</td><td>ADC + DWI → T2W</td><td>21.59 (+0.61)</td><td>0.7140 (+0.0240)</td><td>0.1620 (-0.0160)</td><td>0.0662 (−0.0048)</td></tr><tr><td>Institutional</td><td>D</td><td> $\mathrm { T 2 W + D W I } + \mathrm { A D C }  \mathrm { T 1 W }$ </td><td>25.06 (+0.48)</td><td>0.7960 (+0.0200)</td><td>0.1130 (-0.0120)</td><td>0.0378 (−0.0032)</td></tr><tr><td>Institutional</td><td>E</td><td> $\mathrm { T 2 W + D W I + T 1 W  A D C }$ </td><td>27.82 (+0.46)</td><td>0.8790 (+0.0180)</td><td>0.0810 (-0.0100)</td><td>0.0351 (−0.0029)</td></tr><tr><td>Institutional</td><td>F</td><td> $\mathrm { T 2 W + A D C } + \mathrm { T 1 W }  \mathrm { D W I }$ </td><td>24.53 (+0.45)</td><td>0.8600 (+0.0190)</td><td>0.0900 (-0.0110)</td><td>0.0573 (-0.0037)</td></tr><tr><td>Institutional</td><td>G</td><td> $\mathrm { D W I } + \mathrm { A D C } + \mathrm { T 1 W }  \mathrm { T 2 W }$ </td><td>22.31 (+0.63)</td><td>0.7380 (+0.0250)</td><td>0.1490 (-0.0170)</td><td>0.0617 (-0.0053)</td></tr></table>

Extended Data Table 3. Clinical reader-study and diagnostic-validation summary. Reader scores were obtained in 1,000 held-out cases; diagnostic discrimination was evaluated in a separate 200-case set.
<table><tr><td>Domain</td><td>Endpoint</td><td>Comparison or statistic</td><td>Result</td></tr><tr><td>Reader study</td><td>Overall image quality</td><td>Reference / MSCNet / strongest baseline</td><td>4.71 / 4.32 / 3.23</td></tr><tr><td>Reader study</td><td>Anatomical fidelity</td><td>Reference / MSCNet; secondary descriptive endpoint</td><td>4.67 / 4.29</td></tr><tr><td>Reader study</td><td>Lesion conspicuity</td><td>Reference / MSCNet; lesion-containing cases</td><td>4.46 / 4.07</td></tr><tr><td>Reader study</td><td>Diagnostic confidence</td><td>Reference / MSCNet / strongest baseline</td><td>4.76 / 4.38 / 3.31</td></tr><tr><td>Reader study</td><td>Artefact absence</td><td>Reference / MSCNet / strongest baseline</td><td>4.82 / 4.56 / 3.41; P &lt; 0.001</td></tr><tr><td>Reader agreement</td><td>Inter-reader agreement</td><td>ICC; weighted κ</td><td>0.87; 0.83</td></tr><tr><td>Reader non-inferiority</td><td>Overall quality</td><td>DWI / ADC / T2W / T1W</td><td>Supported / supported / supported / not demonstrated</td></tr><tr><td>PI-RADS</td><td>Exact and within-one agreement</td><td>Reference-based versus MSCNet-based reads; n = 1, 000</td><td>77.8%; 95.6%; weighted κ = 0.871</td></tr><tr><td>Diagnostic validation</td><td>csPCa discrimination</td><td>Reference / MSCNet / baseline</td><td>AUC 0.860 / 0.841 / 0.797</td></tr><tr><td>Diagnostic validation</td><td>AUC difference (95% CI)</td><td>MSCNet-reference / baseline-reference</td><td>-0.019 [-0.048, +0.010] / -0.063 [-0.094, -0.032]</td></tr><tr><td>External validation</td><td>Completion / diagnostic</td><td>Three hospitals; 186 cases (A/B/C: 72/64/50)</td><td>SSIM 0.791; LPIPS 0.124; MAE 0.061; AUC 0.823</td></tr><tr><td>Diagnostic validation</td><td>Sensitivity / specificity</td><td>Reference / MSCNet / baseline</td><td>86/82%; 84/81%; 77/75%</td></tr><tr><td>Diagnostic validation</td><td>PPV / NPV</td><td>Reference / MSCNet / baseline</td><td>71/92%; 69/91%; 61/86%</td></tr><tr><td>Workflow assessment</td><td>Reading time</td><td>Before versus after assistance</td><td>5.1 to 3.4 min; -34%</td></tr><tr><td>Workflow assessment</td><td>Diagnostic confidence</td><td>Before versus after assistance</td><td>3.4 to 4.0; +0.6</td></tr><tr><td>Workflow assessment</td><td>Usability category</td><td>Must rescan / caution / diagnostic-ready</td><td>7% / 17% / 76%</td></tr></table>