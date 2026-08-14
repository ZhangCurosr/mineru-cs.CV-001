# Evaluation of Clinically Steerable Retinal Image Generation from Foundation Model Latent Spaces

Zuzanna A. Wakefield-Skórniewska<sup>1[0009−0000−9050−383X]</sup> and Bartłomiej W. Papież<sup>1[0000−0002−8432−2511]</sup>

Big Data Institute, Nufield Department of Population Health, University of Oxford, United Kingdom

zuzanna.skorniewska@ndph.ox.ac.uk

bartlomiej.papiez@bdi.ox.ac.uk

Abstract. Medical foundation models learn latent representations of clinically meaningful phenotypes, yet their ability to support controllable image generation remains largely unexplored. We evaluate four retinal foundation models within the representation tokenizer framework and examine whether demographic and clinical information encoded in latent representations from foundation models is preserved during synthetic image generation. We show that generated representations and images faithfully inherit phenotype information when evaluated within their originating foundation models, consistently outperforming conventional latent difusion on multiple downstream prediction tasks. However, these gains largely disappear when evaluated using classifiers trained on real images, revealing a previously uncharacterised synthetic-to-real representation gap. These findings demonstrate that foundation-model latent spaces provide a powerful substrate for controllable retinal synthesis while highlighting the need to better align synthetic representations with real-image distributions.

Keywords: Colour Fundus Imaging · Difusion Model · Foundation Model · Steerable Image Generation

## 1 Introduction

The recent advances in self-supervised learning (SSL) have driven the development of foundation models, which are trained on vast unlabelled datasets to learn transferable representations that support a wide range of downstream tasks. In medical imaging, foundation models have emerged across modalities including brain Magnetic Resonance Imaging [12, 20], chest radiography [18, 11], or retinal imaging [22, 9, 21, 15]. Within the field of oculomics, retinal foundation models (RFM) have shown remarkable performance in both retinal [22, 9, 15, 21] and systemic health disease prediction [22], with learned representations often exhibiting stronger associations with clinical outcomes than conventional handcrafted retinal biomarkers [23].

Beyond feature extraction, the learned representations may provide a structured latent space for controllable medical image synthesis. A central challenge in medical image generation is learning latent spaces that capture biologically plausible variability while remaining steerable along clinically meaningful axes of variation. Conventional difusion models [17], however, are optimized for image reconstruction quality rather than the organization of diagnostically relevant features, and their latent spaces are therefore not explicitly structured around clinical phenotypes [16]. We therefore hypothesize that difusion within RFM latent spaces preserves clinically meaningful phenotypic structure more faithfully than conventional latent difusion operating in generic VAE latent spaces.

To test this hypothesis, we evaluate four RFMs as latent spaces for controllable fundus image synthesis. We investigate whether demographic and clinical information encoded in RFM representations is retained in generated fundus images and compare this with conventional latent difusion. We show that RFM latent spaces preserve clinically meaningful phenotypic information more efectively under foundation model-based evaluation. However, these gains diminish when synthetic images are evaluated using classifiers trained on real data, revealing a synthetic-to-real representation gap that highlights an important challenge for clinically reliable image synthesis.

## 2 Methods

![](images/9a2b96b053bb10c27d66884c002624858473135622882cb7e3d2239aa2b4b005.jpg)

(a)  
![](images/97d74091984e0d56dbc6bd2c299c7e2ab2052a190e2512df580d494fc9688f29.jpg)  
Fig. 1. Overview of the proposed framework. (a) A generative decoder is trained to reconstruct VAE-compressed image latents conditioned on fine-tuned retinal foundation model representations. (b) A representation generator trained on outputs (representations) from the frozen foundation model with demographic and clinical conditioning for controllable generation.

## 2.1 Representation Tokenizer

To enable image synthesis, we employed the Representation Tokenizer (RepTok) framework [5], a two-stage generative approach that performs difusion directly in the representation space of a pretrained RFM using a lightweight, attentionfree architecture. In the first training stage (see Fig.1a), a generative decoder is trained conditioned on feature representations from a fine-tuned foundation model (SSL Encoder), whose backbone remains frozen while only its final layer is optimised jointly with the decoder. The decoder, a DiT-B model [4], receives this feature representation via token concatenation and is trained to denoise colour fundus images under a flow matching objective [10]. Prior to denoising, all images are compressed into spatially-reduced latents using a pretrained variational autoencoder (VAE) [14], following the original RepTok formulation. We retain the original cosine-similarity regularisation term, which encourages fine-tuned representations to remain close to their original embeddings while supporting pixel reconstruction learning.

In the second training stage (Fig. 1b), a 1D MLP-Mixer [19] is trained via flow matching to generate CLS-token embeddings from subject-specific demographic and clinical metadata. Classifier-free guidance [8] is incorporated by randomly dropping the conditioning inputs during the training, encouraging the model to learn both condition-specific and condition-invariant representations. At inference, these generated representations are then used to steer controllable image synthesis.

## 2.2 Retinal Foundation Models

We evaluated RepTok using four RFMs spanning vision-only and vision-language models. Among the vision-only models, we employed RETFound [22], a ViT-L [3] encoder trained with masked image reconstruction [6] on 904,170 hospitalsourced fundus images, and PRETI [9], a ViT-B encoder trained with masked reconstruction conditioned on demographic variables (age and sex) on 1,017,549 fundus images. Among the vision-language models, we employed FLAIR [15], which combines a ResNet-50 [7] image encoder with BERT-encoded [2] condition descriptions under a CLIP-style contrastive training objective [13], trained on 288,307 publicly available fundus images; and URFound [21], a vision-language model using a ViT-B image encoder trained jointly on colour fundus and optical coherence tomography (OCT) images via masked image and language reconstruction, paired with BERT encoder for condition descriptions, trained on 180,000 publicly available retinal datasets. For image-generation fine-tuning, we optimise the CLS tokens of RETFound (1,024 parameters), URFound and PRETI (768 parameters each), and for FLAIR we instead optimise its own projection layer of 2048×512 (corresponding to approximately 1M parameters). We compare the four tested RepTok variants with a latent difusion baseline that employs the same frozen VAE and a DiT-B as the difusion backbone.

## 2.3 Data

We trained and evaluated all models on 256×256 colour fundus images from the UK Biobank [1]. For the first training stage (Fig.1a), we used 83,141 images, including both left- and right-eye images, focusing the task primarily on learning faithful image reconstruction. For the second training stage (Fig.1b), we used a subject-unique subset of 50,773 images, to learn the patient-specific representation conditioned on demographic and clinical variables. Conditioning variables included age, sex, body mass index (BMI), current hypertension status, as well as 3-year onsets of myocardial infarction (MI), stroke and chronic obstructive pulmonary disease (COPD). For evaluation, we used a held-out test set of 11,780 unique subjects, with no subject overlap between the training and test data sets.

## 3 Results

![](images/06df71bdc0b299a83c23f2a44a9347fa148b7b379da9b5945a13c553f459fd55.jpg)  
Fig. 2. Real (first column) and synthetic fundus images generated by latent difusion and RepTok conditioned on current hypertension and 3-year onset of chronic obstructive pulmonary disease (COPD), myocardial infarction (MI), and stroke.

## 3.1 Generation and Reconstruction Fidelity

Table 1 summarizes reconstruction and generation quality using reconstruction Fréchet Inception Distance (rFID; images from generative decoder) and generation FID (gFID; images generated using tokens from representation generator). All RFMs achieved comparable reconstruction fidelity with low rFID values. However, generated image quality, measured by gFID, converged at values slightly higher than for the baseline latent difusion model. Among the RepTok variants, FLAIR achieved the best generative performance (lowest gFID), potentially owing to its larger number of trainable parameters, but exhibited reduced reconstruction quality, as measured by peak signal-to-noise ratio (PSNR), structural similarity index (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS), consistent with the reconstruction–generation trade-of reported in the original RepTok study.

Table 1. Reconstruction and generation quality metrics, evaluated on 10,000 real and 10,000 synthetic test images. The real baseline was computed using two independent sets of 10,000 real images.
<table><tr><td>Generation Method</td><td>rFID ↓ PSNR ↑ SSIM ↑ LPIPS ↓ gFID ↓</td><td></td><td></td><td></td><td></td></tr><tr><td>Real</td><td></td><td></td><td></td><td></td><td>0.61</td></tr><tr><td>Latent Diffusion</td><td></td><td></td><td></td><td></td><td>24.13</td></tr><tr><td>RepTok (RETFound) 3.29</td><td></td><td>30.37</td><td>0.86</td><td>0.18</td><td>34.79</td></tr><tr><td>RepTok (PRETI)</td><td>3.41</td><td>29.94</td><td>0.86</td><td>0.18</td><td>38.70</td></tr><tr><td>RepTok (FLAIR)</td><td>3.35</td><td>23.66</td><td>0.82</td><td>0.22</td><td>32.01</td></tr><tr><td>RepTok (URFound)</td><td>3.38</td><td>30.55</td><td>0.86</td><td>0.18</td><td>40.53</td></tr></table>

Table 2. Kernel Inception Distance (KID) reported as mean (std) between real and synthetic images for current hypertension and 3-year onset of myocardial infarction (MI), stroke, and chronic obstructive pulmonary disease (COPD). Lower scores indicate greater similarity between the synthetic and real image distributions.
<table><tr><td>Characteristic</td><td>Real</td><td>Latent Diffusion RETFound</td><td></td><td>PRETI</td><td>FLAIR</td><td>URFound</td></tr><tr><td>Hypertension</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Yes (n=3942) 0.000 (0.000)</td><td></td><td></td><td></td><td></td><td>0.035 (0.004) 0.024 (0.004) 0.031 (0.004) 0.027 (0.003) 0.032 (0.004)</td><td></td></tr><tr><td>MI (3y)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Yes (n=34)</td><td></td><td></td><td></td><td></td><td>0.000 (0.000) 0.026 (0.000) 0.024 (0.000) 0.023 (0.000) 0.024 (0.000) 0.029 (0.000)</td><td></td></tr><tr><td>Stroke (3y)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Yes (n=15)</td><td></td><td></td><td></td><td></td><td>0.001 (0.000) 0.038 (0.000) 0.038 (0.000) 0.047 (0.000) 0.014 (0.000) 0.030 (0.000)</td><td></td></tr><tr><td>COPD (3y)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Yes (n=18)</td><td></td><td>-0.004 (0.000) 0.029 (0.000) 0.036 (0.000) 0.034 (0.000) 0.038 (0.000) 0.028 (0.000)</td><td></td><td></td><td></td><td></td></tr></table>

Despite higher gFID values, qualitative inspection (Fig. 2) shows that all RepTok models produce realistic retinal anatomy, including under rare clinical conditions. Similarly, Kernel Inception Distance (KID) computed on demographic subsets (Table 2) yield comparable results for RepTok, latent difusion, and real data, indicating that both approaches maintain high visual fidelity under conditional generation.

## 3.2 Evaluation on clinical and demographic phenotypes

We first assessed the phenotypic information encoded by each RFM using representations extracted from real images on a range of demographic and general health prediction tasks (Fig. 3; blue bars) and 3-year systemic health onset prediction tasks (Fig. 4; blue bars). All RFM achieved strong performance for sex, age, and hypertension prediction, with URFound and PRETI generally performing best, while BMI prediction remained more challenging across all models. For 3-year onset prediction, all models achieved AUROCs of between 0.6 and 0.7, although FLAIR consistently underperformed the remaining RFM, particularly for stroke and myocardial infarction.

![](images/46e6b739940910e0554d92c215fce17a6a016ce887313678fad221897d76138f.jpg)

![](images/08035b9043d1ec808d37933009329ed0f2c9fb28c28c998b9e81a895ea7cbaf4.jpg)

![](images/45957d4942d18f6bfd59e262d5c079c649899fc473e266d5c059f5862a594e47.jpg)

![](images/f9a0c152eae0734e1001295aa78d933d5b0637f73e4b140bebcda0e212531e2f.jpg)  
Fig. 3. Prediction performance on demographic and hypertension tasks using foundation model representations from real images (blue), tokens from synthetic images (yellow), and synthetic images (purple). Overall, token from foundation model, tokens from synthetic images, and synthetic images all encode demographic information well, outperforming the baseline latent difusion model.

![](images/42cca5c9b179ef54a5e25ec1e9a6fc5d09aea643bcacf170d512a6997ab9e598.jpg)

![](images/54fadfceab44c57af4cfda2c53dee9a31e91fe28fa0bc93b68721da329470cdb.jpg)

![](images/971976778ab82c6acce52789049e44cd2ecc589d1e178c018e43043649622c7d.jpg)  
Fig. 4. Prediction performance for 3 year onset of systemic conditions using foundation model representations from real images (blue), tokens from synthetic images (yellow), and synthetic images (purple).

We next evaluated whether synthetic tokens from the representation generator preserved the conditioned phenotypic information (Figs. 3, 4; yellow bars). As shown, synthetic tokens achieved near-ceiling performance across all conditioned attributes, substantially outperforming representations extracted from real images. BMI remained the most challenging task but was still predicted considerably more accurately than from real-image representations. This strong performance is expected, as the representation generator is explicitly conditioned on the target demographic and clinical variables.

Finally, we evaluated whether phenotypic information was preserved after decoding synthetic representations into fundus images (Figs. 3, 4; purple bars). RepTok images achieved near-ceiling performance for sex and age prediction and generally outperformed real-image tokens for hypertension, BMI, and systemic disease onset prediction. URFound consistently achieved the strongest performance, particularly for disease onset prediction, while FLAIR and PRETI largely matched the performance of real-image representations. Compared with a latent difusion baseline evaluated using URFound, RepTok achieved stronger myocardial infarction prediction performance across all models, with the difusion model performing near chance. URFound was also the only model to consistently outperform latent difusion across all remaining onset prediction tasks. For sex, age, and hypertension, latent difusion achieved performance comparable to real UR-Found tokens, suggesting that the synthetic images preserve at least as much relevant information as the original data.

These results indicate that RFM latent spaces preserve clinically meaningful phenotypic information throughout both representation and image generation. However, the improved compatibility between synthetic images and their originating RFM also suggests that these representations may remain partially model-specific, motivating the external evaluation presented in the following section.

## 3.3 External Validation

Table 3. External validation of phenotype preservation in synthetic images using independently trained ResNet32 classifiers on real fundus images.
<table><tr><td>Imaging</td><td>Age (R2)</td><td></td><td>Sex (Acc.) Hypertension (Acc.)</td></tr><tr><td>Real Data</td><td>0.623</td><td>0.725</td><td>0.717</td></tr><tr><td>Reconstructed</td><td>0.402</td><td>0.695</td><td>0.698</td></tr><tr><td>Latent Diffusion</td><td>0.376</td><td>0.645</td><td>0.717</td></tr><tr><td>RETFound</td><td>0.265</td><td>0.576</td><td>0.709</td></tr><tr><td>PRETI</td><td>0.170</td><td>0.637</td><td>0.714</td></tr><tr><td>FLAIR</td><td>0.279</td><td>0.598</td><td>0.698</td></tr><tr><td>URFound</td><td>0.541</td><td>0.587</td><td>0.718</td></tr></table>

We next evaluated phenotype preservation using ResNet32 classifiers trained exclusively on real images to predict age $( R ^ { 2 } )$ , sex, and hypertension. Incidence outcomes were omitted due to insuficient sample sizes for reliable classifier training. As shown in Tab. 3, the classifiers achieved strong performance on real test data (first row). Reconstructed images (second row) yielded comparable performance for sex and hypertension, but age prediction dropped substantially, with an absolute reduction in explained variance of approximately 20% compared with the real images. This suggests that the classifier may exploit spurious confounding imaging artifacts present in real data that are subsequently removed during reconstruction.

Across synthetic images, the superiority of RepTok observed in the internal evaluation largely disappeared, with latent difusion generally showing stronger conditioning adherence than all RepTok models except URFound. URFound slightly outperformed latent difusion for hypertension and achieved substantially better age preservation (+16.5 percentage points in explained variance). Overall, these findings reveal a gap between internal and external evaluation: although synthetic images retain clinically relevant information when assessed within the synthetic domain, these gains are substantially attenuated when evaluated using classifiers trained on real data. This discrepancy may reflect both domain shift between synthetic and real images and diferences in the features emphasized by foundation models and conventional ResNet classifiers. Notably, the marked reduction in age prediction performance observed for reconstructed real images suggests that external classifier performance is itself sensitive to image transformations and should therefore be interpreted with caution.

## 4 Discussion

In this work, we evaluated whether RFM latent spaces preserve clinically meaningful phenotypic information during controllable image generation. We show that synthetic images inherit demographic and clinical information encoded by RFM, enabling phenotype-preserving image synthesis across multiple clinically relevant attributes. However, these advantages are substantially attenuated under external evaluation with classifiers trained on real images, revealing a synthetic-to-real representation gap that limits downstream transferability.

Among the evaluated RFM, URFound consistently achieved the strongest performance, outperforming latent difusion in both the internal and external evaluations, whereas FLAIR generally performed worst. FLAIR’s weaker performance may be partly explained by its CLIP-based training objective, which was also reported to underperform in the original RepTok study [5]. In contrast, URFound’s strong performance relative to RETFound and PRETI suggests that multimodal pretraining may provide representations that are better suited to controllable image synthesis than vision-only pretraining. Interestingly, URFound achieved these results despite being trained on substantially fewer retinal images than RETFound or PRETI, indicating that pretraining strategy may be at least as important as dataset scale for this task.

Future work should focus on improving the alignment between synthetic and real image representations, enabling synthetic images to retain the clinically meaningful phenotypic information encoded by foundation models while remaining compatible with independently trained models. Bridging this synthetic-toreal representation gap would allow synthetic and real images to be used more interchangeably for downstream clinical tasks. Promising directions include learning representations that are both clinically informative and domain invariant, as well as developing generative objectives that better preserve predictive information across model families.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

Acknowledgement. This research has been conducted using data from UK Biobank with access provided through application 80521. ZAWS is supported by departmental funding from the Nufield Department of Population Health, University of Oxford.

## References

1. Bycroft, C. et al.: The UK Biobank resource with deep phenotyping and genomic data. Nature 2018 562:7726 562(7726), 203–209 (10 2018). https://doi.org/10.1038/s41586-018-0579-z, https://www.nature.com/articles/s41586-018-0579-z

2. Devlin, J. et al.: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL HLT 2019 - 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies - Proceedings of the Conference 1, 4171–4186 (10 2018), https://arxiv.org/pdf/1810.04805

3. Dosovitskiy, A. et al.: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021 - 9th International Conference on Learning Representations (10 2020), https://arxiv.org/pdf/2010.11929

4. Esser, P. et al.: Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. Proceedings of Machine Learning Research 235, 12606–12633 (3 2024), https://arxiv.org/pdf/2403.03206

5. Gui, M. et al.: Adapting Self-Supervised Representations as a Latent Space for Eficient Generation (10 2025), https://arxiv.org/pdf/2510.14630

6. He, K. et al.: Masked Autoencoders Are Scalable Vision Learners. Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition 2022-June, 15979– 15988 (11 2021). https://doi.org/10.1109/CVPR52688.2022.01553, https://arxiv.org/pdf/2111.06377

7. He, K. et al.: Deep Residual Learning for Image Recognition. Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition 2016-December, 770–778 (12 2015). https://doi.org/10.1109/CVPR.2016.90, https://arxiv.org/abs/1512.03385v1

8. Ho, J., Salimans, T.: Classifier-Free Difusion Guidance (7 2022), https://arxiv.org/pdf/2207.12598

9. Lee, Y. et al.: PRETI: Patient-Aware Retinal Foundation Model via Metadata-Guided Representation Learning. Lecture Notes in Computer Science 15960 LNCS, 523–533 (2026). https://doi.org/10.1007/978-3-032-04927-8\_50, https://link.springer.com/chapter/10.1007/978-3-032-04927-8\_50

10. Lipman, Y. et al.: Flow Matching for Generative Modeling. 11th International Conference on Learning Representations, ICLR 2023 (10 2022), https://arxiv.org/pdf/2210.02747

11. Ma, D.A. et al.: A fully open AI foundation model applied to chest radiography. Nature 2025 643:8071 643(8071), 488–498 (6 2025). https://doi.org/10.1038/s41586- 025-09079-8, https://www.nature.com/articles/s41586-025-09079-8

12. Mazher, M., Parker, G.J.M., Alexander, D.C.: Towards generalisable foundation models for brain MRI. npj Imaging 2026 (5 2026). https://doi.org/10.1038/S44303- 026-00176-5, https://www.nature.com/articles/s44303-026-00176-5

13. Radford, A. et al.: Learning Transferable Visual Models From Natural Language Supervision. Proceedings of Machine Learning Research 139, 8748–8763 (2 2021), https://arxiv.org/pdf/2103.00020

14. Rombach, R. et al.: High-Resolution Image Synthesis with Latent Diffusion Models. In: Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition. vol. 2022-June (2022). https://doi.org/10.1109/CVPR52688.2022.01042

15. Silva-Rodríguez, J. et al.: A Foundation Language-Image Model of the Retina (FLAIR): encoding expert knowledge in text supervision. Medical Image Analysis 99, 103357 (1 2025). https://doi.org/10.1016/J.MEDIA.2024.103357, https://www.sciencedirect.com/science/article/pii/S1361841524002822

16. Skórniewska, Z., Papież, B.W.: Exploring the efectiveness of deep features from domain-specific foundation models in retinal image synthesis. In: Annual Conference on Medical Image Understanding and Analysis. pp. 293–305. Springer (2025)

17. Song, J., Meng, C., Ermon, S.: DENOISING DIFFUSION IMPLICIT MODELS. In: ICLR 2021 - 9th International Conference on Learning Representations (2021)

18. Tiu, E. et al.: Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning. Nature Biomedical Engineering 2022 6:12 6(12), 1399–1406 (9 2022). https://doi.org/10.1038/s41551-022-00936-9, https://www.nature.com/articles/s41551-022-00936-9

19. Tolstikhin, I. et al.: MLP-Mixer: An all-MLP Architecture for Vision. Advances in Neural Information Processing Systems 29, 24261–24272 (5 2021), https://arxiv.org/pdf/2105.01601

20. Wu, Y. et al.: BrainDINO: A Brain MRI Foundation Model for Generalizable Clinical Representation Learning (4 2026), https://arxiv.org/pdf/2604.27277

21. Yu, K. et al.: UrFound: Towards Universal Retinal Foundation Models via Knowledge-Guided Masked Modeling. Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics) 15012 LNCS, 753–762 (2024). https://doi.org/10.1007/978-3-031-72390- 2\_70, https://link.springer.com/chapter/10.1007/978-3-031-72390-2\_70

22. Zhou, Y. et al.: A foundation model for generalizable disease detection from retinal images. Nature 2023 622:7981 622(7981), 156–163 (9 2023). https://doi.org/10.1038/s41586-023-06555-x, https://www.nature.com/articles/s41586-023-06555-x

23. Zhou, Y. et al.: AutoMorph: Automated Retinal Vascular Morphology Quantification Via a Deep Learning Pipeline. Translational Vision Science & Technology 11(7), 12–12 (7 2022). https://doi.org/10.1167/TVST.11.7.12, https://doi.org/10.1167/tvst.11.7.12