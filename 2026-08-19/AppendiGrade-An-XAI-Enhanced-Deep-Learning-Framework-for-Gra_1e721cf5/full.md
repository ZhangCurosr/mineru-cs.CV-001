# AppendiGrade: An XAI-Enhanced Deep Learning Framework for Grading Appendicitis in Ultrasound with Gaussian Blur and Grad-CAM

Fahad Ahammed

Computer Science and Information Technology

Ca’ Foscari University of Venice

Venice, Italy

fahadahbd@gmail.com

Md Tahsin   
Computer Science and Engineering   
East West University   
Dhaka, Bangladesh   
tahsin30899@gmail.com

Omar Faruq Shikdar

Md. Nawab Yousuf Ali   
Computer Science and Engineering   
East West University   
Dhaka, Bangladesh   
nawab@ewubd.edu

Computer Science and Engineering East West University Dhaka, Bangladesh omorfaruk549@gmail.com

Navid Zaman

Department of Digital Business

and Innovation

Tokyo International University

Tokyo, Japan

navidzaman.xyz@gmail.com

Golam Sorwar   
Computer Science and Engineering   
Southern Cross University   
Lismore, Australia   
golam.sorwar@scu.edu.au

Abstract—Appendicitis is one of the most common abdominal emergencies worldwide and requires prompt diagnosis and treatment to prevent life-threatening conditions. However, accurately differentiating complicated cases, such as perforation or abscess formation from uncomplicated appendicitis remains a significant clinical challenge. Among other methods, ultrasound is a safer and more cost-efficient diagnostic technique because of a lack of radiation exposure. In this research, an advanced system capable of automatically detecting complicated appendicitis from ultrasound images was developed. A dataset consisting of 4679 ultrasound images with 5 classes perforated, abscess, acute, appendicolith, and normal was used for the proposed model training and testing. Four pretrained deep learning models DenseNet201, InceptionV3, ConvNextTiny, and VGG19, have been employed for detecting and classifying complicated appendicitis. In the initial configuration, InceptionV3 achieved the second highest accuracy, with a value of 69.21%. Owing to suboptimal performance with raw images, further optimization techniques, including image preprocessing, hyperparameter tuning, model fine-tuning, and image sharpening, were applied. These enhancements significantly improved the model’s performance, with an accuracy of 95.58% for InceptionV3. The model performance is then explained with gradient-weighted class activation mapping (Grad-CAM), which creates a heatmap of the regions responsible for the model’s prediction of the infected areas. This could make crosschecking with experts much easier.

Index Terms—Complicated Appendicitis, Ultrasound Image, Data Preprocessing, Image Classification, Grad-CAM

## I. INTRODUCTION

One of the most common abdominal emergencies in the world is appendicitis [1]. From 1990-2019, the incidence of appendicitis increased by 63.55% worldwide [2]. Early and accurate diagnosis is critical for timely treatment, but traditional imaging methods are time-consuming, require expert radiologists, and involve significant training and cost [3].

There are mainly four types of complicated appendicitis cases that can occur. Acute appendicitis is sudden and severe inflammation of the appendix [4]. Perforated appendicitis occurs when infection causes a hole, allowing spread to the abdomen [5]. Appendicitis abscess involves pockets of infection formed as a complication [6]. Appendicolith is a calcified deposit or stone-like object that forms within the appendix [7].

This study aims to address previous gaps by developing a deep learning model for detecting different types of complicated appendicitis from a comprehensive dataset of 4,679 ultrasound images across five classes (acute, perforated, abscess, appendicolith, and normal). Four pretrained models (DenseNet201, InceptionV3, ConvNextTiny, and VGG19) were employed. The models were initially trained on raw images, followed by extensive image preprocessing (Gaussian blur and unsharp masking), layer freezing, and hyperparameter tuning. Grad-CAM was integrated to generate heatmaps highlighting affected areas.

The remainder of this paper is structured as follows. Section 2 discusses the previous study. Section 3 presents the dataset description, preprocessing methodology, and model implementation results. Section 4 compares results across models and before and after optimization. Section 5 concludes with remarks and future work.

## II. LITERATURE REVIEW

Artificial intelligence-based automated diagnostic systems can provide an efficient solution by reducing reliance on radiologists while offering instant results. Most current research focuses primarily on CT and MRI, with several limitations. Byun achieved 85.5% accuracy but was limited to confirmed cases and CT imaging [8]. Walid achieved an AUC of 0.868 but required extensive training data [9]. Rajpurkar introduced AppendiXNet with 72.5% accuracy, though the dataset was too small [10]. Park validated a CNN model with CT data but was limited by a small dataset [11]. Mijwil tested multiple models, achieving 83.75% accuracy but faced interpretability challenges [12]. Pati achieved 92.15% accuracy with AdaBoost but lacked model interpretability [13]. Akbulut employed Catboost with 88.2% accuracy but was limited by retrospective data [14]. Marcinkevics achieved 81.9% accuracy but faced poten-ˇ tial sampling bias [15]. Akmes¸e achieved 95.31% accuracy but was limited to medical records [16]. Although research has used CT and MRI, most studies have been conducted on small datasets, and no research has integrated explainable AI for model interpretability. Ultrasound is the safest option because it does not involve radiation exposure. Poortman compared logistic regression and random forest with CT and sonography images, achieving 83% and 78% accuracy, but lacked advanced ML models [17]. Alelyani achieved 85.7% sensitivity with ultrasound but faced challenges with patient variability [18]. Erdas deployed five pretrained models, with EfficientNetV2 achieving 96% accuracy, but it can detect only acute appendicitis and does not account for other types [19].

On the basis of the above discussion, existing studies using ultrasound for appendicitis diagnosis are limited by small datasets, a lack of explainable AI, and a focus on binary or at most three-class classification, leaving a gap in detecting all major forms of complicated appendicitis.

## III. RESEARCH METHODOLOGY

A dataset of four types of complicated appendicitis was collected. The models were first trained using raw images, but the results were not good enough. Therefore, image preprocessing, layer freezing, and hyperparameter tuning were applied to improve performance. Grad-CAM was then used to explain the model’s decisions

## A. Dataset Description

The dataset encompasses 4679 prelabelled ultrasound images from Roboflow [20], categorized into five classes: abscess, appendicolith, perforated, acute, and normal. A certified physician reviewed and cross-checked the annotations to ensure label accuracy. This dataset is unique in its size, diversity, and balanced distribution across all five classes, providing a valuable opportunity to analyze four distinct types of complicated appendicitis, representing a significant advancement over existing datasets

## B. Dataset Pre-processing

Initially, all images were resized to 200 ∗ 200 pixels to eliminate unnecessary details. During normalization, pixel values were scaled to facilitate faster convergence. Augmentation techniques, such as rotation and flipping, were optionally applied to increase dataset diversity. The model trained on raw images performed poorly due to speckle noise and poor tissue contrast. To enhance performance, Gaussian blur was first applied to reduce noise, followed by an unsharp mask to sharpen images. This combination improves edge detection by reducing noise and enhancing contrast. If Gaussian blur is applied to an original image $I ( x , y )$ to create a smoothed version, denoted as $I _ { - } s m o o t h ( x , y )$ , it can be expressed as:

![](images/5c26a3b84a1400e2d71d25af865d3fcaa7061ad39f589df1280ed1d284cafd68.jpg)  
Fig. 1. Sample distribution of images in each class.

$$
I _ { - } s m o o t h ( x , y ) = ( I * G _ { - } \sigma ) ( x , y )\tag{1}
$$

$$
G _ { - } \sigma ( x , y ) = \frac { 1 } { 2 \pi \sigma ^ { 2 } } * e ^ { - \frac { 1 } { 2 } \frac { ( x - x _ { n } ) ^ { 2 } } { t } }\tag{2}
$$

The smoothed images are generated by convolving I(x,y) with a Gaussian kernel. The smoothed images are subtracted from the original to create a mask highlighting edges:

$$
M a s k ( x , y ) = I ( x , y ) - I _ { - } s m o o t h ( x , y )\tag{3}
$$

Finally, the mask is added to the original image to obtain the sharpened image:

$$
I \_ s h a r p ( x , y ) = I ( x , y ) + a m o u n t * M a s k ( x , y )\tag{4}
$$

![](images/c33142e4c8753dd337d07433d2e10ff18fbf89734c6f939e9330d2a8553f2ff6.jpg)  
Fig. 2. Image Sharpening via Gaussian blur and an unsharp mask.

To achieve a comprehensive evaluation of the models, the dataset was split into 70% for training, 15% for validation, and 15% for testing via stratified sampling. This splitting strategy is standard practice when deep learning models are used [21]. The benefit of stratified sampling is that the class distribution remains the same across all newly created subsets after splitting.

## C. Pretrained models

Four pretrained models were employed: DenseNet201, InceptionV3, ConvNextTiny, and VGG19. DenseNet201’s hierarchical feature extraction is helpful for complex ultrasound images, while InceptionV3’s multiscale features are effective for multiclass classification. ConvNextTiny, a lightweight model, was selected to assess resource-efficient performance, and VGG19 was chosen as a benchmark for its simple architecture.

## D. Model fine tuning and hyperparameter optimization

Each deep learning model had specific layers frozen during fine-tuning: 469 of 706 layers for DenseNet201 and 139 of 310 for InceptionV3. Freezing combinations were tested via grid search. Hyperparameters (learning rate, patience, epochs, batch size, optimizer) were optimized via random search. The Adam optimizer was used. Fig. 3 is a parallel coordinate plot showing relationships between L2 regularization, learning rate, and performance metrics to identify optimal hyperparameter combinations for better generalization.

![](images/79ba70054bdc92c5eccc707f81745fd21c07848d308217aa1238b99391ebaa67.jpg)  
Fig. 3. Hyperparameter tuning combination of DenseNet201

Table 1 on the other hand presents the results of hyperparameter tuning iterations for the Inception V3 model. Both illustrate variations in the learning rate and L2 regularization, along with the corresponding validation loss for each iteration. The primary objective was to identify the optimal combination of hyperparameters that minimizes validation loss, thus improving overall model performance.

TABLE I  
HYPERPARAMETER TUNING ITERATION FOR THE INCEPTIONV3 MODEL
<table><tr><td rowspan=1 colspan=1>Iteration</td><td rowspan=1 colspan=1>Learning Rate</td><td rowspan=1 colspan=1>L2 Regularization</td><td rowspan=1 colspan=1>Validation loss</td></tr><tr><td rowspan=1 colspan=1>00</td><td rowspan=1 colspan=1>0.0007</td><td rowspan=1 colspan=1>0.0035000</td><td rowspan=1 colspan=1>0.28976</td></tr><tr><td rowspan=1 colspan=1>01</td><td rowspan=1 colspan=1>0.0007</td><td rowspan=1 colspan=1>0.0010000</td><td rowspan=1 colspan=1>0.25492</td></tr><tr><td rowspan=1 colspan=1>02</td><td rowspan=1 colspan=1>0.0001</td><td rowspan=1 colspan=1>0.0010000</td><td rowspan=1 colspan=1>0.18773</td></tr><tr><td rowspan=1 colspan=1>03</td><td rowspan=1 colspan=1>0.0005</td><td rowspan=1 colspan=1>0.0085000</td><td rowspan=1 colspan=1>0.24111</td></tr><tr><td rowspan=1 colspan=1>04</td><td rowspan=1 colspan=1>0.0005</td><td rowspan=1 colspan=1>0.0035000</td><td rowspan=1 colspan=1>0.27150</td></tr><tr><td rowspan=1 colspan=1>05</td><td rowspan=1 colspan=1>0.0007</td><td rowspan=1 colspan=1>0.0085000</td><td rowspan=1 colspan=1>0.28386</td></tr><tr><td rowspan=1 colspan=1>06</td><td rowspan=1 colspan=1>0.0005</td><td rowspan=1 colspan=1>0.0060000</td><td rowspan=1 colspan=1>0.28386</td></tr><tr><td rowspan=1 colspan=1>07</td><td rowspan=1 colspan=1>0.0001</td><td rowspan=1 colspan=1>0.0035000</td><td rowspan=1 colspan=1>0.19674</td></tr><tr><td rowspan=1 colspan=1>08</td><td rowspan=1 colspan=1>0.0005</td><td rowspan=1 colspan=1>0.0010000</td><td rowspan=1 colspan=1>0.24710</td></tr><tr><td rowspan=1 colspan=1>09</td><td rowspan=1 colspan=1>0.0001</td><td rowspan=1 colspan=1>0.0085000</td><td rowspan=5 colspan=1>0.196840.307200.562610.236850.21428</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>0.0003</td><td rowspan=1 colspan=1>0.0085000</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>0.0007</td><td rowspan=1 colspan=1>0.0060000</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>0.0001</td><td rowspan=1 colspan=1>0.0060000</td></tr><tr><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>0.0003</td><td rowspan=1 colspan=1>0.0060000</td></tr><tr><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>0.0003</td><td rowspan=1 colspan=1>0.0010000</td><td rowspan=1 colspan=1>0.26137</td></tr></table>

## E. Gradient-weighted class activation mapping

In this study, explainable AI (XAI) Fig. 4 is used to enhance model interpretability using Grad-CAM, which highlights regions of interest in medical images and provides insights into which features influenced the model’s predictions for different types of appendicitis. This approach improved transparency and facilitated understanding of the factors contributing to its decisions. For a given input image, I, and a class score $y ^ { c }$ for class $^ { \mathrm { c } , }$ the gradient of the class score with respect to the feature maps $A ^ { k }$ of a specific layer k is computed as: $\frac { \partial y ^ { c } } { \partial A ^ { k } }$ . The global average of the gradients across all the spatial locations is

$$
\alpha _ { c } ^ { k } = \frac { 1 } { Z } \sum _ { i , j } \frac { \partial y _ { c } } { \partial A _ { i j } ^ { k } }\tag{5}
$$

Class activation map

$$
{ \cal L } _ { \mathrm { G r a d - C A M } } = \mathrm { R e L U } \left( \sum _ { k } \alpha _ { c } ^ { k } A ^ { k } \right)\tag{6}
$$

![](images/9a603697e608bc95620d487fa77d55dc81874cb2a568f3a942f875a5bc45babd.jpg)  
Fig. 4. Explainable AI (XAI) technique Gradient-weighted Class Activation Mapping (Grad-CAM)

## IV. EXPERIMENTAL RESULTS

Table 2 presents the performance results of each model before optimization when the models were trained with only raw images. Among the four models, DenseNet201 achieved the

TABLE II  
RESULTS COMPARISON OF DIFFERENT MODELS BEFORE OPTIMIZATION
<table><tr><td>Model</td><td>Precision</td><td>F1-Score</td><td>Recall</td><td>Accuracy</td></tr><tr><td>DenseNet201</td><td>0.7248</td><td>0.7169</td><td>0.7208</td><td>0.7208</td></tr><tr><td>InceptionV3</td><td>0.6988</td><td>0.6892</td><td>0.6921</td><td>0.6921</td></tr><tr><td>ConvNeXtTiny</td><td>0.6722</td><td>0.6031</td><td>0.6103</td><td>0.6103</td></tr><tr><td>VGG19</td><td>0.5159</td><td>0.2851</td><td>0.3586</td><td>0.3586</td></tr></table>

highest performance, with an accuracy of 72.08%, closely followed by InceptionV3 at 69.21%. Their precision, recall, and F1 scores were 72.48%, 72.08%, 71.69% and 69.88%, 69.21%, 68.92%, respectively. ConvNextTiny achieved 61.03% accuracy, while VGG19 performed the worst at 35.86%. Due to these suboptimal results, we applied image sharpening, fine-tuning, and hyperparameter tuning. These optimization techniques were applied only to the two best models to reduce computational cost. After optimization, DenseNet201 accuracy increased from 72% to 94%, with precision, recall, and F1 score all exceeding 90% presented in Table 3.

Similarly, Table 4 presents how InceptionV3’s performance improved after optimization, with the improvement being even greater than that of DenseNet201. Its accuracy increased from

CLASSIFICATION REPORT OF DENSENET201 AFTER OPTIMIZATION  
TABLE III
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>F1-score</td><td rowspan=1 colspan=1>Support</td></tr><tr><td rowspan=2 colspan=1>AbscessAcuteAppendicolithNormalPerforated</td><td rowspan=2 colspan=1>0.960.900.920.970.95</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.97</td><td rowspan=2 colspan=1>665657569675654</td></tr><tr><td rowspan=1 colspan=1>0.940.970.870.95</td><td rowspan=1 colspan=1>0.920.940.920.95</td></tr><tr><td rowspan=1 colspan=1>accuracymacro average</td><td rowspan=1 colspan=1>0.940.94</td><td rowspan=1 colspan=1>0.940.94</td><td rowspan=1 colspan=1>0.940.94</td><td rowspan=1 colspan=1>32203220</td></tr><tr><td rowspan=1 colspan=1>weighted average</td><td rowspan=1 colspan=3>0.94</td><td rowspan=1 colspan=1>3220</td></tr></table>

69% to 96%. Accuracy was not the only metric that improved, precision, recall, and F1 score all rose above 90%, increasing from approximately 69%.

TABLE IV  
CLASSIFICATION REPORT OF INCEPTIONV3 AFTER OPTIMIZATION
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>F1 score</td><td rowspan=1 colspan=1>Support</td></tr><tr><td rowspan=2 colspan=1>AbscessAcuteAppendicolithNormalPerforated</td><td rowspan=2 colspan=1>0.960.940.980.960.94</td><td rowspan=1 colspan=1>0.98</td><td rowspan=2 colspan=1>0.970.930.970.960.95</td><td rowspan=2 colspan=1>332328284337327</td></tr><tr><td rowspan=1 colspan=1>0.920.970.960.96</td></tr><tr><td rowspan=1 colspan=1>accuracymacro average</td><td rowspan=1 colspan=1>0.960.96</td><td rowspan=1 colspan=1>0.960.96</td><td rowspan=1 colspan=1>0.960.96</td><td rowspan=1 colspan=1>16081608</td></tr><tr><td rowspan=1 colspan=1>weighted average</td><td rowspan=1 colspan=3>0.96</td><td rowspan=1 colspan=1>1608</td></tr></table>

Fig. 5 shows each model’s accuracy progression over training epochs. While both DenseNet201 and InceptionV3 achieved high training accuracy, InceptionV3 demonstrated more stable and consistent performance. DenseNet201 exhibited significant fluctuations in validation accuracy and loss, indicating potential overfitting and instability. In contrast, InceptionV3 displayed smoother convergence with less unpredictability in validation metrics, suggesting better generalizability and reliability with fewer signs of overfitting. Table 5 compares model performance, showing InceptionV3 achieved 99.38% training accuracy and 95.58% test accuracy, while DenseNet201 achieved 99.47% training and 93.94% test accuracy.

TABLE V  
COMPARISON OF THE ACCURACY OF THE INCEPTIONV3 AND DENSENET201 MODELS AFTER OPTIMIZATION
<table><tr><td>Model</td><td>Train Accuracy</td><td>Test Accuracy</td></tr><tr><td>DenseNet201</td><td>99.47%</td><td>93.94%</td></tr><tr><td>InceptionV3</td><td>99.38%</td><td>95.58%</td></tr></table>

InceptionV3’s superior test accuracy indicates better generalizability. This is attributed to its multiscale feature extraction, which handles complex ultrasound textures effectively, and its fewer parameters, enabling better generalization on unseen data. Additionally, InceptionV3’s architecture is inherently better at capturing features from sharpened, noise-reduced images from the comparison, we can clearly conclude that InceptionV3 is the best model. Now, let us examine how optimization improved the model’s performance by comparing its performance before and after optimization. The impact of optimization on the performance of InceptionV3 is evident in the training and validation loss and accuracy curves Figs. 6-7.

![](images/bed51442fbdf944f4d024d33bb19a310516eaf63529ed5ecc223aaf05ff7aa36.jpg)

![](images/6268a4dd9cadc7d3ffc7290fe9ec6e8a79f253de289fa041eadf79aa1e9b62d5.jpg)  
Fig. 5. DenseNet201 and InceptionV3 models training curves after optimiza tion.

![](images/eb445998825df2df4d0a0032a62433c027a843b29d2b015f6fda36d7c0a734d2.jpg)

![](images/49d122a3df628cb5e10edda111468b7508c832f7c9f3b667b6512f4d453d6d39.jpg)  
Fig. 6. Accuracy comparison of InceptionV3 before and after optimization

Before optimization, a clear disparity in performance was evident, with low validation accuracy and high validation loss compared with training metrics. After optimization, the difference became negligible, with training and validation accuracy and loss closely aligned. The comparative analysis in Fig. 8 highlights a significant improvement in InceptionV3 performance across all classes after training with enhanced images. For the abscess class, precision, recall, and F1 score increased from below 90% (precision 61%) to 96%, 98%, and 97%, respectively. Similar improvements were observed for the other classes. For the acute class, precision, recall, and F1 score improved from 71%, 59%, and 65% to 94%, 92%, and 93%, respectively. The most dramatic change was in the appendicolith class, where metrics increased from 65%, 55%, and 60% to 98%, 97%, and 97%, representing a 30-37% improvement.

![](images/c83ab87b78e6e8bd0773796187a2365966868199f876aea48cba544754f328b8.jpg)  
Fig. 7. Loss comparison of InceptionV3 before and after optimization

![](images/4c0ee288e4bccf50573a6b606c2564e9fe4a9af0a4197f55a6a7681484cf3ac0.jpg)  
Fig. 8. Comparison between before and after optimization for InceptionV3

The remaining two classes showed similar improvements. For the perforated class the precision increased by 26%, the recall increased by 31%, and the F1 score increased by 30%. For the normal class, the change was minimal, as the model’s performance for this class was already the best among all classes. After enhancement, the precision, recall, and F1 score all improved by 14-15%. The confusion matrix Fig.

9 for InceptionV3 after optimization highlights its improved accuracy. Only 7 abscess classes, 27 acute classes, 9 appendicolith classes, 14 normal classes and 14 perforated classes of ultrasound images were misclassified by the inception model. Overall, the model correctly classified 1,537 out of 1,608 ultrasound images.

![](images/97e824b9f674f209e4a7b53af80ce56e8ef3bdabc08e0a0dfb657a0453ec060a.jpg)  
Fig. 9. Confusion matrix for InceptionV3 after optimization

![](images/6ba7aeda6378a56accfb65d7e4481f469494c396204f7f9aa4e1c3ccd625f706.jpg)  
Fig. 10. ROC curve for InceptionV3 after optimization

The ROC curve Fig. 10 for our best-performing model, InceptionV3, is shown. The model achieved a perfect AUC score of 1.0 for four classes and a near-perfect 0.99 for the remaining class, demonstrating excellent performance across all classes. Model performance on the test dataset is shown in Fig. 11.

Grad-CAM was integrated to visualize model predictions. The heatmap Fig. 12 highlights regions influencing classification decisions: red indicates the greatest influence, yellow moderate, and green to blue low to very low importance. These visualizations confirm that the areas visually distinguishing each type of complicated appendicitis are primarily responsible for the model’s correct predictions.

![](images/61a13a609e95aed481e6ca8a5563a69db0fa1180f77fc4b1194070c2060e33c5.jpg)  
Fig. 11. Model prediction on test data for each class

![](images/bc1661b8518e0fec748f2b10b4a98367f9c5755820d87b8b69db092aa2f246d6.jpg)

![](images/20e91db99a3a1bc0ff3e7748c1cef733c946a3f80db839902abe53c9082942c5.jpg)  
Fig. 12. Grad-CAM heatmap for each type of complicated appendicitis for InceptionV3

## V. CONCLUSION

This study investigates different deep learning models to achieve the best results for detecting various types of appendicitis. Using default model setups on raw images yielded suboptimal results, with overfitting and significant disparities between training and testing accuracy. By leveraging multiple optimization techniques, substantial improvements were achieved, with InceptionV3 accuracy increasing from 69.21% to 95.38%. To improve interpretability, Grad-CAM was incorporated to provide visual details into the model’s decision-making process. The system has the potential to be integrated into healthcare, assisting physicians by highlighting areas of concern in ultrasound images and offering probable classification. The current study has several limitations: the dataset may not fully represent real-world variations, image sharpening can obscure details, heatmaps have not been verified by doctors, and the model has not been tested in diverse hospital settings. To address these, future research will focus on refining the system through clinical trials and pilot testing in collaboration with hospitals.

## REFERENCES

[1] A. Petroianu, “Acute appendicitis – propedeutics and diagnosis,” InTech eBooks, 2012.

[2] Y. Yang et al., “The global burden of appendicitis in 204 countries and territories from 1990 to 2019,” Clinical Epidemiology, vol. 14, pp. 1487–1499, 2022.

[3] M. M. Fischmann and F. Bellolio, “Acute appendicitis,” in Wiley, pp. 457–471, 2023.

[4] S. Malik, J. E. Harris, R. Vasdev, and A. T. Rezcallah, “Perforated appendicitis presenting as a soft tissue infection of the thigh: An example of successful non-operative management,” Cureus, 2023.

[5] L. P. H. Andersen, T. Pedersen, and J. Rosenberg, “Femoral abscess after acute appendicitis,” European Geriatric Medicine, 2012.

[6] S. P. Dhital et al., “Appendicoliths: The neglected stones,” Journal of Patan Academy of Health Sciences, 2017.

[7] J. Byun, S. Park, and S. M. Hwang, “Diagnostic algorithm based on machine learning to predict complicated appendicitis in children using CT, laboratory, and clinical features,” Diagnostics, vol. 13, no. 5, p. 923, 2023.

[8] W. A. Al, I. D. Yun, and K. J. Lee, “Reinforcement learning-based automatic diagnosis of acute appendicitis in abdominal CT,” arXiv preprint, 2019.

[9] M. M. Mijwil and K. Aggarwal, “A diagnostic testing for people with appendicitis using machine learning techniques,” Multimedia Tools and Applications, vol. 81, no. 5, pp. 7011–7023, 2022.

[10] P. Rajpurkar et al., “AppendiXNet: Deep learning for diagnosis of appendicitis from a small dataset of CT exams using video pretraining,” Scientific Reports, vol. 10, 2020.

[11] A. Pati et al., “Predicting pediatric appendicitis using ensemble learning techniques,” Procedia Computer Science, vol. 218, pp. 1166–1175, 2023.

[12] S. Akbulut et al., “Prediction of perforated and nonperforated acute appendicitis using machine learning-based explainable artificial intelligence,” Diagnostics, vol. 13, no. 6, p. 1173, 2023.

[13] R. Marcinkevicsˇ et al., “Using machine learning to predict the diagnosis, management and severity of pediatric appendicitis,” Frontiers in Pediatrics, vol. 9, 2021.

[14] O. F. Akmes¸e<sup>¨</sup> et al., “The use of machine learning approaches for the diagnosis of acute appendicitis,” Emergency Medicine International, 2020.

[15] J. J. Park et al., “Convolutional-neural-network-based diagnosis of appendicitis via CT scans in patients with acute abdominal pain,” Scientific Reports, vol. 10, 2020.

[16] P. Poortman et al., “Comparison of CT and sonography in the diagnosis of acute appendicitis: A blinded prospective study,” American Journal of Roentgenology, vol. 181, no. 5, pp. 1355–1359, 2003.

[17] M. Alelyani et al., “Evaluation of ultrasound accuracy in acute appendicitis diagnosis,” Applied Sciences, vol. 11, no. 6, p. 2682, 2021.

[18] A. R. O. Erdas¸ and C¸ . B. Erdas¸, “Computer-aided diagnosis of acute<sup>¨</sup> appendicitis in pediatric patients using deep learning models on abdominal ultrasound images,” in Proc. Int. Academic Conf. Global Education, Teaching and Learning, Dec. 2024.

[19] “abdoDetection Computer Vision Project,” Roboflow Universe, 2024.

[20] P. Baheti, “Train test validation split: How to & best practices,” V7 Labs, 2021.

[21] M. E. Heilbrun et al., “Assessing the training costs and work of diagnostic radiology residents using key performance indicators,” Academic Radiology, vol. 27, no. 7, pp. 1025–1032, 2020.