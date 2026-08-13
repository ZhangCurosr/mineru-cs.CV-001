# A comparison of CNN architectures for Alzheimer’s disease detection in single-view MRI scans

Hiram Zuniga<sup>a</sup>, Ulises Orozco-Rosas<sup>a,∗</sup>, and Kenia Picos<sup>a</sup>

<sup>a</sup>CETYS Universidad, Ave. CETYS Universidad No. 4, Fracc. El Lago, C.P. 22210, Tijuana, Baja California, Mexico

## ABSTRACT

Alzheimer’s disease is a leading cause of death with no cure. Therefore, early detection is critical to slow progression and preserve quality of life. Diagnosis relies on medical history, cognitive tests, physical exams, and MRI brain scans, making deep learning suitable for Alzheimer’s classification. This work proposes a benchmark that evaluates ten diferent convolutional neural network (CNN) architectures (including ResNet, DenseNet, MobileNet, EficientNet, and VGG family models) under the same held-out test split protocol. A two-stage transfer learning and full fine-tuning pipeline is introduced to perform training using a class-balanced subset (3,900 images) derived from the OASIS medical imaging dataset, comprising 86,437 single-view MRI brain scans labeled into four classifications of Alzheimer’s disease: Non-Demented, Very Mild Dementia, Mild Dementia, and Moderate Dementia. The best results were achieved by VGG16, with a 0.9637 validation accuracy and a 0.9533 test accuracy score. A key finding documented in this work is the dificulty of classifying the transition from Non-Demented to Very Mild Demented stages, observed consistently across all ten architectures.

Keywords: computer-aided diagnosis, transfer learning, deep learning, medical imaging, OASIS dataset, benchmark

## 1. INTRODUCTION

Alzheimer’s disease (AD) has been the seventh leading cause of death in the United States in recent years,<sup>1</sup> and is the most common cause of dementia in the elderly,<sup>2</sup> afecting nearly one-half of Americans older than 85 years old<sup>3</sup> and 22% of all persons aged 50 and above.<sup>4</sup> AD impacts the lives of nearly 7.2 million people older than 65 in the US, and this number could grow to 13.8 million in 2060 (almost double). AD has caused nearly 120,122 deaths in the US, only in 2022. Between 2000 and 2022, the reported deaths from AD have increased by more than 142%, so not only is AD not going anywhere but is increasing the number of deaths and the number of people afected at a rapid rate.<sup>1</sup>

Despite advances in Alzheimer’s research, there is still no current treatment to cure the disease, no clearly effective disease-modifying treatments,<sup>5,</sup> <sup>6</sup> and no pharmacologic agents can reverse the progression.<sup>3,</sup> <sup>6</sup> Alzheimer’s disease is a fatal degenerative dementing disorder, which starts with memory deterioration that progresses to debilitating loss of mental and physical faculties.<sup>7,</sup> <sup>8</sup> Early detection of this disease have been seen to be relevant, because you can treat reversible causes of dementia<sup>9</sup> and implement strategies to reduce or prevent further progression of the disease’s, improving quality of life,<sup>6,</sup> <sup>9</sup> some of the treatments and recent medicine advances has shown that donepezil is efective in stabilizing or slowing progressive decline in cognition, function, and behavior.<sup>10</sup>

Built upon these ideas, early diagnosis is important because it can represent a big clinical advantage in initiating treatment early in the course of the disease,<sup>10</sup> and it also gives the patient time to make choices or plan ahead for the future and their carers.<sup>6</sup> Currently, AD diagnosis has been dependent mainly on clinical suspicion based on the patient’s or caregiver concerns,<sup>9</sup> medical history,<sup>11</sup> psychometric scales, Neuropsychological testing to evaluate patient cognitive function, and neuroimaging as an auxiliary diagnostic tool; these include magnetic resonance imaging (MRI) to detect brain structural changes.<sup>12</sup> This makes deep learning and medical imaging a suitable aid for diagnosis using these MRI brain scans.

![](images/606f9420a61d95839cd6fe39c8d428781a3b06e0d50128cf594572b2c5350b07.jpg)  
Figure 1: Overview of the benchmark pipeline using the OASIS dataset proposing a two-stage training protocol. Snowflake and flame icons denote frozen and trainable components, respectively.

Deep learning techniques have been widely explored for Alzheimer classification using MRI brain scans.<sup>13–15</sup> This work presents a benchmark that evaluates ten diferent Convolutional Neural Network (CNN) architectures under the same protocol. To evaluate these neural networks, this study introduces a two-stage transfer learning and full-tuning pipeline: 5 epochs are used to train the head classifier of the CNN with a frozen backbone, and the remaining 20 epochs are used for full fine-tuning on the CNN architecture with an unfrozen backbone.

The dataset used in this study is the OASIS medical dataset,<sup>16</sup> comprising 86,437 single-view MRI brain scan images, labeled into four classes: Mild Dementia, Moderate Dementia, Very Mild Dementia, and Non Demented. During training across these diferent CNN architectures, a key finding was noted: all models struggle to diferentiate between the Non Demented and the Very Mild Dementia classes. Additionally, an overview of the described pipeline is presented in Fig. 1.

The key contributions of this work are described as follows:

• A benchmark that evaluates ten diferent CNN architectures under the same evaluation protocol, comprising: the same split, the same image preprocessing, the same augmentation, sampler, batch size, and epochs.

• A two-stage transfer learning and full fine-tuning pipeline, using 5 epochs for transfer learning with the frozen backbone and 20 epochs for fine-tuning with an unfrozen backbone.

The remainder of the paper is organized as follows: Section 2 provides an overview of the materials and methods, describing the dataset, preprocessing, the architectures, and so on. Section 3 presents the results from running the experiments and provides a technical discussion on them. Section 4 introduces the conclusion and discusses future research directions.

## 2. EXPERIMENTAL METHODOLOGY

This section presents all of the methods used to obtain the results. Additionally, it introduces the dataset and the split, the CNN architectures benchmarked in the study, the evaluation metrics, and the experimental setup for reproducibility.

## 2.1 Dataset and partitioning

This work uses the OASIS Alzheimer detection dataset<sup>16</sup> for training and evaluation, having 86,437 single-view MRI brain scan images. This dataset is based on 416 subjects aged 18 to 96 years, and images are divided into 4 classes: Mild Dementia with 5,002 images, Moderate Dementia with 488 images, Very Mild Dementia with 13,725 images, and Non Demented with 67,222 images. Based on the number of images per class, a class imbalance problem is notable, more specifically for the Moderate Dementia class, which has only 488 images compared to the 67,222 images in the Non-Demented class. Dataset details are shown in Table 1. Additionally, some samples from the dataset are shown in Fig. 2.

Table 1: Class distribution of the OASIS single-view MRI dataset.
<table><tr><td>Class</td><td>Images</td><td>Share (%)</td></tr><tr><td>Mild Dementia</td><td>5,002</td><td>5.79</td></tr><tr><td>Moderate Dementia</td><td>488</td><td>0.56</td></tr><tr><td>Non-Demented</td><td>67,222</td><td>77.77</td></tr><tr><td>Very Mild Dementia</td><td>13,725</td><td>15.88</td></tr><tr><td>Total</td><td>86,437</td><td>100.00</td></tr></table>

![](images/affaf8dc28566dbfe278fb82cde4354a6541816ba6da0568f480eba668e93f69.jpg)  
(a) Mild

![](images/d21a2c43603ab4991ee7b7a1aa6b59c7efa417dec8f42c2c8137ef4fa9983949.jpg)  
(b) Moderate

![](images/e47ecc60260867f2fcb3baac9e3aa34bfb2991442166e27e43f0a8397416cee6.jpg)  
(c) Non-Demented

![](images/820ae8d10bca8b5129134b0517bcf180e3c1a1118ce9105580d50ea09e5123b1.jpg)  
(d) Very mild  
Figure 2: Representative single-view MRI slices from the four OASIS classes: (a) Mild Dementia, (b) Moderate Dementia, (c) Non-Demented, and (d) Very Mild Dementia.

To mitigate the class imbalance efect, this study caps all images to 800 for the training cohort, 200 for validation, and 150 for a held-out test split. The reason to cap the images from the dataset instead of applying techniques like class weighting, random undersampling (RUS), synthetic minority over-sampling technique (SMOTE), and others is that the key objective of this work is to benchmark these ten diferent architectures under single evaluation conditions, rather than maximizing the accuracy value that can be achieved.

Since the natural cap is 488 for the Moderate Dementia class, this study divides these images into 100 for training, 200 for validation, and 150 for testing. The reason is to have original images for the validation and testing splits. Additionally, augmentation is applied to training samples to reach 800 samples for the other classes. Augmented samples are deliberately excluded from the validation and test splits, as the study requires a clean and consistent comparison across architectures. The details on how many images are available during the experiments are available in Table 2.

Table 2: Class-balanced partition of the OASIS single-view MRI dataset used in this work. The training column reports the number of distinct original images; the balanced sampler described in Section 2.2 fixes the number of samples drawn per epoch at 800 per class.
<table><tr><td>Class</td><td>Train (original)</td><td>Train (per epoch)</td><td>Validation</td><td>Test</td></tr><tr><td>Mild Dementia</td><td>800</td><td>800</td><td>200</td><td>150</td></tr><tr><td>Moderate Dementia</td><td>100</td><td>800</td><td>200</td><td>150</td></tr><tr><td>Non-Demented</td><td>800</td><td>800</td><td>200</td><td>150</td></tr><tr><td>Very mild Dementia</td><td>800</td><td>800</td><td>200</td><td>150</td></tr><tr><td>Total</td><td>2,500</td><td>3,200</td><td>800</td><td>600</td></tr></table>

## 2.2 Preprocessing and class balancing

To ensure a consistent evaluation, the same preprocessing pipeline was applied across the 10 CNNs. First, the images are preprocessed to a single grayscale channel; they are resized to 224×224 pixels, converted to tensors, and normalized with a mean value of 0.5 and a standard deviation of 0.5. This normalization was used instead of ImageNet standard normalization since RGB does not apply in this setting, and only one Grayscale channel is used.

A Weighted Random Sampler is introduced with an objective of 3,200 samples, meaning that the Moderate Dementia class is sorted 8 times per epoch, and in each sort, augmentations are applied consisting of random afine with degrees = 10 and color jitter with a brightness value of 0.08 and a contrast of 0.08. All of these new samples are generated on the fly, rather than precomputed and stored beforehand. This enables matching the same 800 images for training the other classes. An important distinction is that augmentation is not applied to the validation and testing splits. This ensures that the evaluation reflects original OASIS data, which is closer to what the model would encounter at inference, instead of artificially altered images.

ImageNet weights expect a three-channel RGB input. Since this classification is performed on single-channel grayscale MRI brain scans, most of the pretrained architectures are not compatible with this input. This work replaces the first convolutional layer with an equivalent layer expecting one input channel instead of three, and its weights are initialized by averaging the pretrained weights across the three RGB channels, as expressed in Eq. (1), where $\mathbf { W } _ { c } ^ { ( 1 ) }$ denotes the pretrained weights of the first convolutional layer corresponding to channel c.

$$
\mathbf { W } _ { \mathrm { g r a y } } ^ { ( 1 ) } = \frac { 1 } { 3 } \sum _ { c = 1 } ^ { 3 } \mathbf { W } _ { c } ^ { ( 1 ) }\tag{1}
$$

This method retains features learned during ImageNet pretraining, rather than reinitializing the first layer with random weights. This procedure was applied uniformly across the ten architectures, adapting only the location where the first convolutional layer resides in each model, as it is defined diferently across architectures.

## 2.3 CNN architectures under comparison

Ten diferent CNN architectures are evaluated to compare performance in the Alzheimer classification task. These diferent CNNs were selected to cover diferent design families such as plain stack, residual, grouped residual, densely connected, compound scaled, and mobile-oriented architectures.<sup>17,</sup> <sup>18</sup> This inclusion of heterogeneous architectures provides a complete comparison across diferent CNNs with diferent purposes. All of these CNNs were initialized with IMAGENET1K V1 weights from torchvision, pretrained on ImageNet,<sup>19</sup> and are introduced in Table 3.

Table 3: CNN architectures evaluated in this work. Head parameter counts correspond to the classification head only, and total parameter counts include the single-channel first convolutional layer.
<table><tr><td>Architecture</td><td>Design family</td><td>Total params</td><td>Head params</td><td>Pretrained head</td></tr><tr><td>VGG1620</td><td>Plain deep stack</td><td>134.3 M</td><td>119,562,244</td><td>Yes</td></tr><tr><td>ResNet1821</td><td>Residual</td><td>11.3M</td><td>132,356</td><td>No</td></tr><tr><td>ResNet3421</td><td>Residual</td><td>21.4M</td><td>132,356</td><td>No</td></tr><tr><td>ResNet5021</td><td>Residual</td><td>24.0M</td><td>525,572</td><td>No</td></tr><tr><td>ResNet10121</td><td>Residual</td><td>43.0M</td><td>525,572</td><td>No</td></tr><tr><td>ResNeXt50-32x4d²2</td><td>Grouped residual</td><td>23.5 M</td><td>525,572</td><td>No</td></tr><tr><td>DenseNet12123</td><td>Densely connected</td><td>7.2M</td><td>263,428</td><td>No</td></tr><tr><td>DenseNet16923</td><td>Densely connected</td><td>12.9M</td><td>427,268</td><td>No</td></tr><tr><td>EfficientNet-B024</td><td>Compound-scaled</td><td>4.3M</td><td>328,964</td><td>No</td></tr><tr><td>MobileNetV3-L25</td><td>Mobile / NAS</td><td>4.2M</td><td>1,235,204</td><td>No</td></tr></table>

As shown in Table 3, the evaluated architectures span a wide range of model sizes, from 4.2M parameters for the mobile and eficient CNNs to 24M parameters for the larger ResNet models. Additionally, ResNet101 and VGG16 have 43M and 134.3M parameters, respectively; these architectures represent larger models, and this benchmark will let us evaluate the diference in computational time and performance between larger and smaller models to assess whether large architectures pose an advantage in this Alzheimer classification task.

Table 4: Shared classification head applied to eight of the ten architectures. The input dimension d is the feature dimension of each backbone (e.g., 512 for ResNet18/34, 2,048 for ResNet50/101 and ResNeXt50). VGG16 and MobileNetV3-L use diferent heads, as described in the text.
<table><tr><td>Layer</td><td>Type</td><td>Output dim.</td></tr><tr><td>1</td><td>Linear (d → 256)</td><td>256</td></tr><tr><td>2</td><td>ReLU</td><td>256</td></tr><tr><td>3</td><td>Dropout (p = 0.5)</td><td>256</td></tr><tr><td>4</td><td>Linear (256 → 4)</td><td>4</td></tr></table>

Eight of the ten architectures receive the same classification head, described in Table 4. Two exceptions apply. VGG16 retains its original pretrained classifier, consisting of two layers of 4096 neurons and only replacing the output layer with one compatible with four classes. This classifier design was kept as it represents 89% of the whole model (Table 3) and replacing it with the classifier described in Table 4 would have discarded the majority of the pretrained model. MobileNetV3-L follows the same structure as the shared head but preserves a 1280-unit hidden layer of its original design.

## 2.4 Two-stage transfer-learning and fine-tuning protocol

This work introduces a two-stage transfer-learning and fine-tuning pipeline for training, applied identically to the ten models to ensure consistent training conditions across all architectures. The first phase is transfer learning: training only the head classifier while keeping the encoder backbone frozen, with a learning rate of $1 \times 1 0 ^ { - 4 }$ for 5 epochs, as this phase is a head warm-up before adapting the full architecture. Phase two is full fine-tuning; after the first 5 epochs, the backbone is unfrozen all at once, and then 20 epochs are applied to train the ful encoder with ${ \mathrm { ~ a ~ } } 1 \times 1 0 ^ { - 5 }$ learning rate, which is lower to reduce the risk of overfitting, since this risk is higher when adapting the full architecture rather than only the head. Both phases use the Adam optimizer and are trained on a GPU. A representative diagram of this two-stage pipeline is shown in Fig 3.

![](images/739782e753caf2bc79aa3028d238434d72dfc585f650927dc342327cb2267722.jpg)  
Figure 3: Two-stage transfer-learning and fine-tuning protocol, applied identically to all ten architectures. Snowflake and flame icons denote frozen and trainable components, respectively.

Training in two stages avoids the risk of overfitting that arises when full fine-tuning is applied from the first epoch. This is especially true with the small number of images used for training (800 images) and the large number of parameters and layers some architectures can reach. Therefore, phase 1 stabilizes the head classifier before entering full fine-tuning within the MRI domain.

## 2.5 Evaluation protocol and metrics

To ensure a consistent comparison between architectures, this work introduces an identical evaluation protocol for CNNs. In preprocessing, the same channel adaptation was used for all ten models. During training, the same sampler, batch size, optimizer, loss function, number of epochs, and learning rate were applied across all ten CNNs. Additionally, the same held-out test split with the same seed was used for evaluation across all ten models. The only thing that changed across these ten architectures was the backbone and that is what makes this evaluation work, and the head exception already declared in the subsection 2.3. More details on these training configurations are available in Table 5.

Table 5: Training configuration, identical for all evaluated architectures.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Input resolution</td><td> $2 2 4 \times 2 2 4 .$  single channel</td></tr><tr><td>Normalization</td><td>mean 0.5, std 0.5</td></tr><tr><td>Batch size</td><td>4</td></tr><tr><td>Samples per epoch</td><td>3,200 (800 per class)</td></tr><tr><td>Phase 1 – epochs</td><td>5 (classification head only)</td></tr><tr><td>Phase 1 – optimizer</td><td>Adam, learning rate  $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Phase 2 – epochs</td><td>20 (all parameters)</td></tr><tr><td>Phase 2 – optimizer Adam, learning rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Loss function</td><td>Cross-entropy</td></tr><tr><td>Model selection</td><td>Highest validation accuracy</td></tr><tr><td>Random seed</td><td>42</td></tr></table>

This work evaluates the models on their best epoch. A checkpoint retains the best-performing parameters on the validation split during training across the 25 epochs. Additionally, the held-out test split was used only for final evaluation; it was not used during training or validation to ensure no data leakage occurred before the final test. This held-out test was only used to report final metrics, and the results did not influence training or reporting decisions to ensure the benchmark holds value.

Metrics reported in the benchmark are global accuracy, the macro- and weighted-average versions of precision, recall, and F1-score, and support, all described in Eq. (2)–(6). For a given class $k , \mathrm { T P } _ { k } , \mathrm { F P } _ { k }$ , and $\mathrm { F N } _ { k }$ denote true positives, false positives, and false negatives; $K = 4$ is the number of classes; $n _ { k }$ is the support of class $k ;$ and $N = \textstyle \sum _ { k } n _ { k }$ is the total number of test samples.

$$
\mathrm { A c c u r a c y } = { \frac { \sum _ { k = 1 } ^ { K } { \mathrm { T P } _ { k } } } { N } }\tag{2}
$$

$$
P _ { k } = { \frac { { \mathrm { T P } } _ { k } } { { \mathrm { T P } } _ { k } + { \mathrm { F P } } _ { k } } } , \qquad R _ { k } = { \frac { { \mathrm { T P } } _ { k } } { { \mathrm { T P } } _ { k } + { \mathrm { F N } } _ { k } } } , \qquad F 1 _ { k } = { \frac { 2 P _ { k } R _ { k } } { P _ { k } + R _ { k } } }\tag{3}
$$

$$
\mathrm { M a c r o } \cdot M = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } M _ { k } , \qquad M \in \{ P , R , F 1 \}\tag{4}
$$

$$
\mathrm { W e i g h t e d - } M = \frac { 1 } { N } \sum _ { k = 1 } ^ { K } n _ { k } M _ { k } , \qquad M \in \{ P , R , F 1 \}\tag{5}
$$

$$
{ \mathrm { S u p p o r t } } _ { k } = n _ { k }\tag{6}
$$

These metrics are only reported on the Macro version, since the four classes have the same quantity of samples (150 images), and weighted and macro metrics represent the same value. All metrics were computed on the held-out test set. Additionally, a confusion matrix was used as a supporting metric, calculated for all models but not reported for each one.

## 2.6 Experimental setup

All of the results were generated under the same environmental setup: a computer equipped with an Intel Core i9 Central Processing Unit (CPU) and an NVIDIA GeForce RTX 5060 Graphics Processing Unit (GPU). Al experiments were performed using Python 3.12.10, with libraries such as PyTorch and torchvision for model definition, pretrained weight loading, and training; scikit-learn for computing evaluation metrics and confusion matrices; and NumPy for deterministic data partitioning. A fixed random seed of 42 was used throughout, so that the class-balanced split described in Section 2.1 is reproducible across all ten architectures.

## 3. EXPERIMENTAL RESULTS

After training all the CNNs under the two-stage transfer-learning and fine-tuning pipeline, they were evaluated through the same evaluation protocol using a held-out test split of 150 images per class (600 images in total). The results of benchmarking these diferent architectures are shown in Table 6.

Table 6: Test-set performance of the ten evaluated architectures under the same protocol, ordered by test accuracy. Precision, recall, and F1 are macro-averaged.
<table><tr><td>Architecture</td><td>Val. acc.</td><td>Test acc.</td><td>Precision</td><td>Recall</td><td>F1</td><td>Errors</td></tr><tr><td>VGG16</td><td>0.9637</td><td>0.9533</td><td>0.9536</td><td>0.9533</td><td>0.9531</td><td>28</td></tr><tr><td>ResNet18</td><td>0.9375</td><td>0.9400</td><td>0.9395</td><td>0.9400</td><td>0.9394</td><td>36</td></tr><tr><td>ResNet34</td><td>0.9525</td><td>0.9400</td><td>0.9397</td><td>0.9400</td><td>0.9397</td><td>36</td></tr><tr><td>ResNet50</td><td>0.9363</td><td>0.9333</td><td>0.9343</td><td>0.9333</td><td>0.9331</td><td>40</td></tr><tr><td>ResNeXt50-32x4d</td><td>0.9388</td><td>0.9250</td><td>0.9285</td><td>0.9250</td><td>0.9258</td><td>45</td></tr><tr><td>ResNet101</td><td>0.9388</td><td>0.9217</td><td>0.9259</td><td>0.9217</td><td>0.9222</td><td>47</td></tr><tr><td>DenseNet121</td><td>0.9300</td><td>0.9167</td><td>0.9188</td><td>0.9167</td><td>0.9169</td><td>50</td></tr><tr><td>EfficientNet-B0</td><td>0.8963</td><td>0.9033</td><td>0.9065</td><td>0.9033</td><td>0.9034</td><td>58</td></tr><tr><td>DenseNet169</td><td>0.9350</td><td>0.9000</td><td>0.9039</td><td>0.9000</td><td>0.9003</td><td>60</td></tr><tr><td>MobileNetV3-L</td><td>0.9163</td><td>0.8817</td><td>0.8882</td><td>0.8817</td><td>0.8827</td><td>71</td></tr></table>

The best-performing architecture was VGG16 on all four metrics (Accuracy: 0.9533, Precision: 0.9536, Recall: 0.9533, F1: 0.9531), committing 28 errors. The best architectures from the ResNet family are ResNet18 (0.9400) and ResNet34 (0.9400), outperforming the larger models in the same family: ResNet50 (0.9333), ResNeXt50- 32x4d (0.9250), and ResNet101 (0.9217). Additionally, the worst-performing CNNs were among the densely connected and eficient or mobile-optimized architectures, with the worst performance reached by MobileNetV3- L (0.8817), which nonetheless remains a competitive result, given that this is the smallest encoder (4.2M parameters). The total diference between the best-performing model and the worst is 0.072 points, which is not a substantial gap. This suggests the two-stage training protocol worked and that the first 5 transfer learning epochs helped the model stabilize before the 20 full fine-tuning epochs.

Additional findings indicate that no model appears to sufer severe overfitting. Most models showed a validation-to-test diference below 0.02, suggesting they generalize well on unseen data. The exceptions are DenseNet169 and MobileNetV3-L, both above 0.03, which are also the two worst-performing models on the heldout test split. Additionally, VGG16 did not overfit regardless of being the largest model (134.3M parameters).

## 3.1 Depth does not improve accuracy within architecture families

An interesting finding made visible after evaluating the ten diferent CNNs is that performance degrades consistently on larger architectures within the same family or design. For example, this is most notable on the ResNet family, where ResNet18 (11.3M parameters) achieves a 0.9400 test accuracy, while ResNet101 (43.0M parameters) achieves 0.9217 test accuracy. ResNet101 has 3.8 times more parameters than ResNet18 and is stil outperformed by the smaller version. These examples are present across all ten CNNs benchmarked and are presented in Table 7.

Table 7: Efect of depth within architecture families. Within each residual and densely-connected family, increasing depth consistently reduced test accuracy.
<table><tr><td>Family</td><td>Variant</td><td>Total params</td><td>Val. acc.</td><td>Test acc.</td><td>Gap (pp)</td></tr><tr><td rowspan="4">Residual</td><td>ResNet18</td><td>11.3M</td><td>0.9375</td><td>0.9400</td><td>-0.25</td></tr><tr><td>ResNet34</td><td>21.4M</td><td>0.9525</td><td>0.9400</td><td>+1.25</td></tr><tr><td>ResNet50</td><td>24.0 M</td><td>0.9363</td><td>0.9333</td><td>+0.30</td></tr><tr><td>ResNet101</td><td>43.0M</td><td>0.9388</td><td>0.9217</td><td>+1.71</td></tr><tr><td rowspan="2">Densely conn.</td><td>DenseNet121</td><td>7.2M</td><td>0.9300</td><td>0.9167</td><td>+1.33</td></tr><tr><td>DenseNet169</td><td>12.9M</td><td>0.9350</td><td>0.9000</td><td>+3.50</td></tr><tr><td rowspan="2">Residual vs. grouped</td><td>ResNet50</td><td>24.0M</td><td>0.9363</td><td>0.9333</td><td>+0.30</td></tr><tr><td>ResNeXt50-32x4d</td><td>23.5M</td><td>0.9388</td><td>0.9250</td><td>+1.38</td></tr></table>

There is no ground-truth reason for these larger models to degrade in performance in a consistent manner on all design families. An attributable factor is the mismatch between model capacity and the available data quantity from this classification task, because larger models can learn more representations and are trained on ImageNet, spanning a million images across a thousand classes, but when, during training, they are introduced to only 800 images per class, these models can overfit or memorize the training set to some extent. This can be seen in Table 7, where test accuracy decreases monotonically with depth in every family, and the largest variant of each family shows a bigger validation-to-test gap than the smallest model. Nonetheless, this is attributed to the gradual degradation of performance between smaller and larger models. The observed gaps indicate a mild tendency toward overfitting that grows with capacity, not severe overfitting, which was not observed in any o the evaluated models.

Building on this, VGG16 is the largest model in terms of parameter count, yet the top performer. This is because 89% of the model parameters are in the classifier, which is pretrained and is not the main component responsible for extracting key features from the images. Consequently, the total parameter count on the backbone of the CNN is 14,713,536 (14M), similar to the ResNet18 architecture (11.3M). Consistent with this, VGG16 has one of the smallest validation-to-test gaps (+1.04 pp) despite its size.

## 3.2 The dificulty in classifying Non-Demented and Very Mild Demented samples

The second key finding in this study was the notable dificulty all models had in classifying Non-Demented and Very Mild Demented samples. This is due to the structural similarity between these two categories; there is not an aggressive change from a non-demented brain to a very mild demented brain, whereas the diference from a non-demented brain to a moderate demented brain is far more pronounced. Additionally, this study works with 2D slices instead of a 3D image, where these distinctions could be more apparent.

The distribution of these errors on the non-demented (ND) and very mild demented (VMD) samples is illustrated in Fig. 4. As shown in this figure, the distribution of mistakes across the ND and VMD classes remains similar across the 10 models. This suggests all architectures struggle with these two classes, and when the error increases, it usually increases in the other 2 classes, but the distribution of misclassifications on ND and VMD classes remains similar.

![](images/f969b92a85b8582d96bf4d58e9a9d8dd88e3756818f3ab5594afb54481acd7f0.jpg)  
Figure 4: Error decomposition across all ten architectures, ordered by test accuracy. Between 51% and 79% of every model’s misclassifications fall on the single Non-Demented / Very Mild Dementia boundary.

To provide a good example, the two confusion matrices of the best- and worst-performing models (VGG16 and MobileNetV3-L) are shown in Fig. 5. The confusion matrices show that both models struggle most with the Non-Demented and Very Mild classes, comprising 78.6% of the errors on VGG16 (22 out of 28 errors) and 50.7% of the errors on MobileNetV3-L (36 out of 71 errors). This also suggests that as the model improves, it reduces other errors but continues to struggle with classifying the ND and VMD classes.

![](images/c5ab360c8cb1191493a15e271755b386d50930dd75cdb78d56f09fa00959590e.jpg)

(b) MobileNetV3-L — 0.8817  
![](images/453395929c688f2d4df678239cbfe20b2ea8c390711cfb151d49d320d81b375c.jpg)  
Figure 5: Confusion matrices on the held-out test set for the best (VGG16) and worst (MobileNetV3-L) models.

## 3.3 Performance versus computational cost

A further analysis of how cost-efective these architectures are is illustrated in Fig. 6. The main finding is that, even though VGG16 achieved the highest performance, it also required the most computational time to train, taking 30 minutes to complete the 25 training epochs. In contrast, ResNet18 took only 7 minutes to train the same 25 epochs and achieved a 0.9400 accuracy score. Compared to the 0.9533 score of the VGG16, ResNet18 is a more cost-efective solution for Alzheimer’s disease classification, given that the gap between these two models is only 0.0133 in accuracy.

![](images/a9e041d8bd3e91d3b4d0f5fb4367736ac4dffea72ae540d828453c77ddd6ea02.jpg)  
Figure 6: Test accuracy versus training time. ResNet18 reaches 98.6% of VGG16’s accuracy in one quarter of the training time, while MobileNetV3-L, designed for eficiency, takes as long as ResNet18 yet scores 0.0583 lower on accuracy.

Another finding was made when comparing ResNet18 and MobileNetV3-L; the latter CNN is optimized for mobile use with short inference and training times. This is true because both CNNs took 7 minutes to train, but MobileNet was the worst-performing CNN, and ResNet18 was the second best-performing CNN. Therefore, even though they are the two fastest architectures, the performance tradeof is notable on ResNet18. These two findings position ResNet18 as a great alternative to VGG16 due to its training time, which would be a great advantage if we were to train on the entire OASIS dataset, comprising 86,437 images.

Additionally, ResNet50 took more time to train (19 minutes) than ResNet18 (7 minutes) and had inferior performance (0.9333 vs. 0.9400), reinforcing the idea discussed in Section 3.1 that deeper models do not necessarily contribute beyond computational cost. The same pattern is visible with DenseNet169 and ResNet101. These results should be interpreted with the understanding that the study used an RTX 5060 GPU to generate the results and serves as a model comparison; other studies may report diferent times due to diferent experimenta environments.

## 3.4 Training discussion

VGG16 is the CNN architecture with the highest performance across the four metrics among the ten architectures. As previously discussed in section 3.2, this model also struggles in diferentiating samples from the Non-Demented class and the Very Mild Dementia class. This is evident in the individual performance table of the VGG16 model presented in Table 8, where the non-demented and very mild dementia classes have the lowest precision, recall, and

F1 scores. Another tendency visible only in this table is the preference to classify patients as Alzheimer positives rather than classify them into a non-demented class when the model is uncertain about the classification. This is shown in the recall metric across the Alzheimer-positive classes: Mild Dementia: 0.9867, Moderate Dementia: 1.0000, and Very Mild Dementia: 0.9467, against a 0.8800 recall value for the Non-demented class. This specific behavior is beneficial in this setting, where sending a person to a doctor for further clinical review is preferred rather than classifying him as non-demented and delaying a doctor’s diagnosis.

Table 8: Per-class test performance of the best model (VGG16).
<table><tr><td>Class</td><td>Precision</td><td>Recall</td><td>F1</td><td>Support</td></tr><tr><td>Mild Dementia</td><td>0.9737</td><td>0.9867</td><td>0.9801</td><td>150</td></tr><tr><td>Moderate Dementia</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>150</td></tr><tr><td>Non-Demented</td><td>0.9362</td><td>0.8800</td><td>0.9072</td><td>150</td></tr><tr><td>Very Mild Dementia</td><td>0.9045</td><td>0.9467</td><td>0.9251</td><td>150</td></tr><tr><td>Macro avg</td><td>0.9536</td><td>0.9533</td><td>0.9531</td><td>600</td></tr></table>

The two-phase training curves of VGG16 are presented in Fig. 7. During phase 1, with the backbone frozen, validation loss is unstable and peaks promptly around epoch 5 alongside accuracy. The transition to full finetuning in phase 2 immediately stabilizes both validation loss and accuracy, which supports the choice of the two-stage training protocol. A notable feature is that validation accuracy continues to improve until epoch 22, where it achieves the best checkpoint with 0.9637 accuracy, even after training accuracy has already reached 1.0000. This suggests the model keeps generalizing after it has memorized the training set, which is why the best checkpoint is selected by validation accuracy rather than using the score at the final epoch.

![](images/ed59e3d1490b89f9f47aef2660a2058033ec21b9ef02d08b5bd2f625b796c84b.jpg)  
(a) Loss

![](images/77870eb6389c1e483f5d1b6f0f70904ff8eaa851da7f5fceaef065957d5917ea.jpg)  
(b) Accuracy  
Figure 7: Two-phase training curves of VGG16 across 25 epochs. Panel (a) shows cross-entropy loss and panel (b) shows accuracy, each reporting the training and validation sets. The shaded region corresponds to Phase 1 (transfer learning), and the remainder to Phase 2 (full fine-tuning). The circled marker in panel (b) indicates the checkpoint selected for evaluation, chosen by the highest validation accuracy.

## 3.5 Limitations

Some limitations should be considered when interpreting these results and are described as follows:

• Slice-level partitioning. The dataset was split at the level of individual 2D slices rather than by subject, so slices from the same scan may appear in both the training and test sets. The relative ranking across architectures produced by the benchmark remains valid, since all models were evaluated on the identical partition.

• Non-uniform classification head. The head is not matched across all ten architectures: VGG16 retains its pretrained classifier while the others use a randomly-initialized head.

• Single cohort. All images used for evaluation come from a single dataset (OASIS). Generalization under domain shift, to images acquired with diferent scanners and acquisition protocols remains untested.

## 4. CONCLUSIONS & FUTURE WORK

This work introduces a benchmark of ten diferent CNN architectures evaluated under the same protocol, consisting of a held-out test split only used to report final results. Training was performed using a proposed two-stage transfer-learning and fine-tuning pipeline, whose objective is to stabilize head adaptation before proceeding to full fine-tuning, potentially avoiding overfitting given the limited size of the training split (3200 images) and the large quantity of parameters in some of the evaluated models. VGG16 was the best-performing architecture, achieving a test accuracy score of 0.9533 on the held-out test split. Additionally, ResNet18 established itself as a strong cost-efective alternative (0.9400 accuracy), taking 22 fewer minutes to train (7.5 vs. 30.1 minutes) while trading of only 0.0133 in accuracy.

Two main findings were visible consistently across the benchmark. First, within each model family, larger variants degraded performance rather than improved test accuracy. This is attributable to a mismatch between model capacity and the limited training data. Second, across all models evaluated, most errors are concentrated on the Non-Demented and Very Mild Dementia classes. This is due to structural similarity in the brain between these two categories, where an aggressive change is not notable. Future work could focus on evaluating these CNNs under a subject-level protocol to eliminate slice-level data leakage, extending the benchmark with Vision Transformers (ViT), or evaluating on an external held-out dataset to quantify performance under domain shift.

## ACKNOWLEDGMENTS

This work was supported by the Coordinaci´on Institucional de Investigaci´on of CETYS Universidad.

## REFERENCES

[1] Alzheimer’s Association, “2025 Alzheimer’s disease facts and figures,” Alzheimer’s & Dementia 21(4), e70235 (2025).

[2] Cummings, J. L. and Cole, G., “Alzheimer disease,” JAMA 287(18), 2335–2338 (2002).

[3] Winslow, B. T., Onysko, M. K., Stob, C. M., and Hazlewood, K. A., “Treatment of alzheimer disease,” American Family Physician 83(12), 1403–1412 (2011).

[4] Gustavsson, A., Norton, N., Fast, T., Fr¨olich, L., Georges, J., Holzapfel, D., Kirabali, T., Krolak-Salmon, P., Rossini, P. M., Ferretti, M. T., Lanman, L., Chadha, A. S., and van der Flier, W. M., “Global estimates on the number of persons across the Alzheimer’s disease continuum,” Alzheimer’s & Dementia 19(2), 658–670 (2023).

[5] Petrella, J. R., “Neuroimaging and the search for a cure for alzheimer disease,” Radiology 269(3), 671–691 (2013).

[6] Rasmussen, J. and Langerman, H., “Alzheimer’s disease - why we need early diagnosis,” Degenerative Neurological and Neuromuscular Disease 9, 123–130 (2019).

[7] Smith, M. A., “Alzheimer disease,” in [International Review of Neurobiology], Bradley, R. J., Harris, R. A., and Jenner, P., eds., 42, 1–54, Academic Press (1998).

[8] Ganguli, M., Dodge, H. H., Shen, C., Pandav, R. S., and DeKosky, S. T., “Alzheimer disease and mortality: A 15-year epidemiological study,” Archives of Neurology 62(5), 779–784 (2005).

[9] de Levante Raphael, D., “The knowledge and attitudes of primary care and the barriers to early detection and diagnosis of Alzheimer’s disease,” Medicina 58(7), 906 (2022).

[10] Gauthier, S. G., “Alzheimer’s disease: the benefits of early treatment,” European Journal of Neurology 12(s3), 11–16 (2005).

[11] Reiss, A. B., de Levante Raphael, D., Chin, N. A., and Sinha, V., “The physician’s Alzheimer’s disease management guide: Early detection and diagnosis of cognitive impairment, Alzheimer’s disease and related dementia,” AIMS Public Health 9(4), 661–689 (2022).

[12] Liu, E., Zhang, Y., and Wang, J.-Z., “Updates in Alzheimer’s disease: from basic research to diagnosis and therapies,” Translational Neurodegeneration 13(1), 45 (2024).

[13] Shaikh, M. R., Jeyabose, A., and Arjunan, R. V., “Deep learning for Alzheimer’s disease: advances in classification, segmentation, subtyping, and explainability,” BioMedical Engineering OnLine 24(1), 150 (2025).

[14] Nagarajan, I. and Lakshmi Priya, G. G., “A comprehensive review on early detection of Alzheimer’s disease using various deep learning techniques,” Frontiers in Computer Science 6 (2025).

[15] Raj, A. S., Gunasundari, C., Senthilkumar, S., and Sivamani, S., “An optimized hybrid deep learning model to detect Alzheimer disease,” Scientific Reports 15(1), 34081 (2025).

[16] Marcus, D. S., Wang, T. H., Parker, J., Csernansky, J. G., Morris, J. C., and Buckner, R. L., “Open access series of imaging studies (oasis): Cross-sectional mri data in young, middle aged, nondemented, and demented older adults,” Journal of Cognitive Neuroscience 19(9), 1498–1507 (2007).

[17] Lopez-Montiel, M., Orozco-Rosas, U., S´anchez-Adame, M., Picos, K., and Ross, O. H. M., “Evaluation method of deep learning-based embedded systems for trafic sign detection,” IEEE Access 9, 101217–101238 (2021).

[18] Lopez-Montiel, M., Orozco-Rosas, U., S´anchez-Adame, M., Montiel, O., Picos, K., and Tapia, J. J., “Trafic sign classification using real-time gpu-embedded systems,” SN Computer Science 7(1), 12 (2025).

[19] Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L., “ImageNet: A large-scale hierarchical image database,” in [IEEE Conference on Computer Vision and Pattern Recognition (CVPR)], 248–255 (2009).

[20] Simonyan, K. and Zisserman, A., “Very deep convolutional networks for large-scale image recognition,” in [3rd International Conference on Learning Representations (ICLR)], (2015). arXiv:1409.1556.

[21] He, K., Zhang, X., Ren, S., and Sun, J., “Deep residual learning for image recognition,” in [IEEE Conference on Computer Vision and Pattern Recognition (CVPR)], 770–778 (2016).

[22] Xie, S., Girshick, R., Doll´ar, P., Tu, Z., and He, K., “Aggregated residual transformations for deep neural networks,” in [IEEE Conference on Computer Vision and Pattern Recognition (CVPR)], 5987–5995 (2017).

[23] Huang, G., Liu, Z., van der Maaten, L., and Weinberger, K. Q., “Densely connected convolutional networks,” in [IEEE Conference on Computer Vision and Pattern Recognition (CVPR)], 2261–2269 (2017).

[24] Tan, M. and Le, Q. V., “EficientNet: Rethinking model scaling for convolutional neural networks,” in [Proceedings of the 36th International Conference on Machine Learning (ICML)], Proceedings of Machine Learning Research 97, 6105–6114, PMLR (2019).

[25] Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., and Adam, H., “Searching for MobileNetV3,” in [IEEE/CVF International Conference on Computer Vision (ICCV)], 1314–1324 (2019).