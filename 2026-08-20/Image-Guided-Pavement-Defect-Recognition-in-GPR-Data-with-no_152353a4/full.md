# Image-Guided Pavement Defect Recognition in GPR Data with novel 3D Deep Learning Architecture

Yuandong Pan, Ph.D.<sup>1</sup>, Linjun Lu, Ph.D.<sup>2</sup>, Mudan Wang, Ph.D.<sup>3</sup>, Florian Noichl<sup>4</sup>, Fan Xue,

Ph.D.<sup>5</sup>, Brian Sheil, Ph.D.<sup>6</sup>, Lavindra de Silva, Ph.D.<sup>7</sup>, André Borrmann, Ph.D.<sup>8</sup>, and Ioannis

Brilakis, Ph.D.<sup>9</sup>

<sup>1</sup>Postdoctoral scholar, School of Engineering, Stanford University, 473 Via Ortega, Stanford

<sup>2</sup>Research fellow, Department of Engineering, University of Cambridge, 7a JJ Thomson Avenue,

Cambridge. Corresponding Email: ll718@cam.ac.uk

<sup>3</sup>Assistant research professor, Department of Engineering, University of Cambridge, 7a JJ

Thomson Avenue, Cambridge

<sup>4</sup>Postdoctoral Scholar, School of Engineering and Design, Technical University of Munich,

Arcisstrasse 21, Munich

<sup>5</sup>Associate professor, Department of Real Estate and Construction, University of Hong Kong, Pok

Fu Lam Rd, Hong Kong

<sup>6</sup>Associate professor, Department of Engineering, University of Cambridge, 7a JJ Thomson

Avenue, Cambridge

<sup>7</sup>Research professor, Department of Engineering, University of Cambridge, 7a JJ Thomson

Avenue, Cambridge

<sup>8</sup>Professor, School of Engineering and Design, Technical University of Munich, Arcisstrasse 21,

Munich

<sup>9</sup>Professor, Department of Engineering, University of Cambridge, 7a JJ Thomson Avenue,

Cambridge

## ABSTRACT

Ground Penetrating Radar (GPR) is a widely adopted non-destructive sensing technology for

subsurface inspection in civil and transportation engineering. Despite its potential for pavement condition assessment, the large-scale application of GPR in automated inspection has two key challenges: the scarcity of annotated real-world datasets and the lack of deep learning models designed for the unique characteristics of 3-Dimensional (3D) GPR data. This study addresses these limitations by firstly introducing a cost-efective data preparation pipeline that integrates orthomosaic Red Green Blue (RGB) imagery with 3D GPR scans to generate annotated 3D GPR datasets. The proposed method uses the aligned segments of RGB and GPR data, using pavement surface images as a reference to transfer labels of surface-visible defects to corresponding GPR segments, enabling eficient large-scale annotation in a real-world dataset collected on a highway section under operation. In addition to the dataset contribution, we propose a specialised 3D Convolutional Neural Network (CNN) architecture incorporating residual connections, mixed convolutional kernel sizes, and both depthwise and channelwise attention mechanisms to enhance feature representation and defect classification. The model is evaluated on binary classification tasks for detecting patch and crack defects in pavement structures. Experimental results demonstrate that the proposed network outperforms baseline architectures across multiple evaluation metrics. Ablation studies further confirm the efectiveness of the designed architectural components. This work contributes a scalable and practical method for real-world dataset generation, along with a novel deep learning framework. It establishes a data foundation for developing more advanced machine learning approaches and benefits the automation and accuracy of GPR-based pavement defect detection in large-scale infrastructure assessment.

## PRACTICAL APPLICATIONS

Maintaining roads and pavements is essential for transportation safety and the long-term durability of infrastructure. Ground Penetrating Radar (GPR) is a widely adopted non-destructive sensing technology for subsurface inspection in civil and transportation engineering. Traditionally, identifying subsurface defects such as cracks or voids using GPR data is time-consuming, requires specialised domain expertise, and often involves techniques like drilling. This study presents a new method that combines GPR data with surface images to identify pavement defects. By using highresolution road imagery to guide the labelling of 3D GPR data, the approach enables the creation of large datasets without manual interpretation of complex GPR scans. A specially designed 3D deep learning model is then trained to recognise defect patterns within this data and outperforms benchmark methods. This paper has three main contributions: a novel data preparation framework that uses an RGB image to annotate GPR data, a 3D real-world annotated dataset, and a novel highperforming deep learning architecture for the data. This brings a significant step toward smarter, data-driven infrastructure management with advanced GPR data analysis.

## INTRODUCTION

Road networks are essential infrastructure for economic growth and societal well-being, forming the backbone of the transportation system for moving both goods and passengers. In 2018, 79% of domestic freight shipments in the United Kingdom were carried by road (Department for Transport, UK b). By 2021, cars, vans, and taxis accounted for 88% of passenger kilometres travelled (Department for Transport, UK a). Highways play a particularly critical role within road networks, carrying substantial trafic volumes. For example, although motorways and "A" roads comprised only 13% of the total road network in the UK by length in 2022, they supported 64% of the total motor vehicle distances travelled (Department for Transport, UK c). Given their importance, highways require regular pavement inspections to ensure timely maintenance and reliable infrastructure. Without proper upkeep, deteriorating highways can lead to a range of serious consequences, including safety hazards, vehicle damage (e.g., from potholes), and trafic congestion. These issues not only pose risks to public well-being but also contribute to economic losses and increased pollutant emissions.

In general, there are three categories of pavement assessment: visual inspection, destructive, and non-destructive testing technologies. Manual visual inspection, particularly for crack detection in images and videos, is widely adopted in industry, while developing automated vision-based methods is a well-researched area. However, such an approach is restricted to detecting visible surface defects, enabling only reactive repairs after apparent deterioration. This reactive approach to pavement maintenance can adversely afect highway functionality, as visible damage is often an indicator of advanced deterioration. To mitigate these impacts, maintenance practices increasingly require methods capable of detecting potential, yet unobservable defects within pavement structures. Such methods can inform targeted maintenance interventions and reduce trafic disruption. While destructive inspection methods (e.g. pavement corings) are often used to evaluate subsurface pavement conditions, they also present clear disadvantages: they damage the pavement structure, require dedicated operational time windows (often necessitating road closures), and involve relatively high costs and labour demands (Rasol et al. 2022).

For these reasons, non-destructive methods, and particularly ground penetrating radar (GPR), are commonly adopted for subsurface inspections of infrastructure (Malihi et al. 2024). GPR is a non-destructive technology that is capable of detecting subsurface features, providing insights beyond the visible surface. It has been employed for a wide range of civil engineering tasks, including the detection of hidden utilities and objects (Ren et al. 2024; Wang et al. 2022), as well as pavement inspection and repair evaluation (Liu et al. 2023c; Liu et al. 2023a).

In road inspection, GPR technology mainly works as a complementary survey combined with more traditional methods, such as visual inspection, drilling, and sampling. Several organizations have established standards to guide the application of GPR for road infrastructure inspection. The American Society for Testing and Materials (ASTM) International published D6432-19 to guide the assessment of subsurface materials using GPR (ASTM International ). In the United Kingdom, National Highways has published guidance documents on the non-destructive testing of highway structures (National Highways, UK b) and procedures for pavement assessment (National Highways, UK a). These guides include detailed requirements and specifications for employing GPR in road and pavement evaluations.

Interpreting GPR data, especially for applications such as detecting pavement structures and assessing subsurface conditions, requires significant expert knowledge and is not an intuitive process. The data analysis phase requires substantial manual input, which inhibits scalability for more routine applications. Additionally, human interpretation introduces variability and the potential for errors, as the analysis is heavily reliant on the expertise of the individual. Consequently, there is an increasing demand for automating the interpretation of GPR data to reduce human efort (Kuchipudi et al. 2022), dependence on expert knowledge, and variability in analysis.

Deep learning methods have been applied in processing data in road digitalization and condition assessment (Lu et al. 2025; Lu et al. 2026). By leveraging models trained on annotated datasets, GPR data can be analyzed to generate predictions based on patterns and knowledge derived from the training data. However, applications of supervised deep learning for interpreting GPR data in large-scale road networks face key challenges, notably the scarcity of annotated datasets.

Synthetic data are used when real data with annotations is expensive to obtain, and have been proven to have their potential to improve the performance (Zhou et al. 2025). Similarly, while synthetic GPR data can be generated using simulation-based approaches, these are limited by their inability to fully capture the complexity and heterogeneity of real-world subsurface conditions, such as variations in material properties and environmental factors (Wu and Sheil 2025). As a result, models trained on simulated data often fail to generalize to real-world applications. In addition, expert ’ground truth’ annotation is prone to variability and errors, while physical sampling is often infeasible for roads in active operation. These limitations highlight the need for more scalable and cost-efective approaches to GPR data annotation and interpretation in practical infrastructure applications.

Therefore, the remaining challenges can be summarised from both data and algorithmic perspectives:

• Lack of real-world benchmark datasets: Most existing approaches have been tested on datasets collected by the respective authors, making it dificult to evaluate these methods consistently. The absence of a standardized, real-world benchmark dataset hinders the ability to compare diferent approaches under identical test conditions.

• Challenges in data annotation: Annotations are essential for most machine learning-based approaches, yet current methods for annotating GPR data present significant issues. Broadly, there are two primary approaches: (1) annotation through expert interpretation and (2) annotation using test samples from traditional experimental techniques e.g. pavement coring. While expert-based annotation is comparatively faster and cheaper, it depends heavily on the interpreter’s experience, introducing variability and subjectivity. These issues are generally less pronounced in the annotation of other data types, such as images. In contrast, annotations derived from physical testing ofer high accuracy and reliability but are labour-intensive, time-consuming, and expensive. Developing eficient and costefective annotation strategies is crucial to addressing these limitations and advancing machine learning applications in GPR data analysis.

• Limited focus on algorithmic development for real-world 3-Dimensional (3D) GPR data: Despite being more informative, 3D GPR data remains underexplored. Generic deep learning architectures are not specifically designed for the characteristics of real-world GPR data, where the longitudinal, lateral, and depth dimensions have diferent physical meanings and noise characteristics. There is a pressing need for deep learning architectures that can directly process real-world 3D GPR data and efectively learn features across these dimensions.

To address the abovementioned challenges of annotating GPR data, we propose a novel approach to generating labels by combining pavement images with corresponding GPR data from the same areas. By using visual information from images to guide the annotation of GPR data, we bypass the need for direct annotation of GPR data itself. Image annotation is more straightforward and generates labelled data more eficiently, which is important to apply GPR technology in large-scale road network applications. To address the gap in algorithmic development for 3D GPR data, we propose a novel deep learning architecture to improve feature extraction from volumetric GPR data collected from real-world operational roads.

The contributions of this paper can be summarized as follows:

• We propose a novel network architecture that integrates 3D convolutional blocks, depth attention mechanisms, and channel attention mechanisms. The proposed architecture achieves robust classification performance on our dataset, demonstrating its efectiveness in process-

ing 3D GPR data.

• We introduce a novel dual-modality dataset consisting of co-registered GPR and Red Green Blue (RGB) image samples collected along the longitudinal road direction from a section of concrete pavement on Highway A14 in the UK.

• Instead of relying solely on traditional methods such as coring or expert-based ground truth labelling when annotating GPR data, we propose using RGB images as a reference to generate proxy labels for spatially corresponding GPR segments associated with surfacevisible pavement defects. Our study shows that deep learning models can learn GPR patterns associated with such defects.

The rest of this paper is organised as follows. Research background is provided in Section 3; The research methodology is introduced in Section 4; The experiments and results are presented in Section 5; conclusions and future work are discussed in Section 6.

## RESEARCH BACKGROUND

GPR operates by exploiting the varying propagation characteristics of electromagnetic waves as they travel through diferent media, which are determined by the media’s dielectric constants. When the signal encounters an interface between two media with significantly diferent dielectric constants, a portion of the signal is reflected while the remainder continues propagating into deeper layers. These reflected signals are then detected by a receiving antenna to identify subsurface material variations (Solla et al. 2021). GPR data can be interpreted either by analysing the reflected signals directly or by reconstructing 2-Dimensional (2D) or 3D images of the subsurface.

## Signal-based methods

Traditional methods, compared with recently developed machine learning approaches, can be broadly divided into two categories: signal-based and image-based methods. In signal-based research, researchers have focused on developing methods to analyze the electromagnetic waves received by GPR antennas. Fundamental techniques for processing GPR data, such as filtering, deconvolution, and time-zero correction, are described in (Jol 2008). Building on these foundational methods, numerous advanced techniques have been proposed and applied in pavement assessment. For instance, regularized deconvolution combined with the L-curve method has been used to determine the thicknesses of various layers in asphalt pavements at a field test site (Zhao et al. 2015). Similarly, in estimating asphalt density of pavement, a correction algorithm based on baseline GPR scans has been proposed and validated through pavement coring (Pengcheng Shangguan and Zhao 2016).

Those methods typically require knowledge and expertise in multiple domains, including signal processing, the operational principles of GPR, and the electromagnetic wave behaviours when propagating through diferent media. Moreover, reliable information extraction often depends on prior knowledge of the scanned targets. For example, Li et al. (2016) employ a Hough transform algorithm for object recognition tasks, which necessitates prior information about the size and orientation of the scanned objects.

## Image-based methods

Image-based methods in GPR are primarily employed for processing B-scan and C-scan data. For detecting subsurface utilities such as pipes and cables, two characteristic patterns—hyperbolic curves and linear segments—are commonly observed in B-scan GPR images (Bruschini et al. 1998). Numerous methods have been developed to identify these patterns. For instance, Dou et al. (2017) proposed a novel clustering algorithm to identify regions of interest, subsequently fitting hyperbolas to these regions. Additionally, the dimensions of buried objects can be derived by analyzing B-scan images; for example, Chang et al. (2009) presented procedures for determining the radius of rebar from B-scan images, validating their approach using several laboratory concrete specimens.

Compared to B-scans, C-scans, representing 3D GPR data generated from a multi-antenna GPR system and composed of multiple B-scans along an orthogonal direction (Klęsk et al. 2015), provide a more intuitive and easily interpretable representation (Luo et al. 2019). However, due to the high computational cost associated with processing C-scan data (Klęsk et al. 2015), many researchers prefer to simplify the problem into a 2D space. Instead of analyzing hyperboloids in

3D, some studies utilize their 2D projections as hyperbolas and implement separate calculations (Zhu and Collins 2005; Frigui et al. 2010). Some studies have also tackled C-scan data directly. For example, Jing and Vladimirova (2017) propose a feature-based algorithm that highlights changes in reflection values between the layers within C-scan data.

Similar to signal-based methods, image-based approaches yield reliable results when carefully calibrated, considering factors such as device specifications and the electromagnetic properties of materials. However, a significant challenge remains: these methods heavily rely on expert knowledge. As noted in Tong et al. (2018), these techniques are not robust in complex scenarios, often requiring human intervention and assistance.

## Learning-based approaches

As a prominent data-driven method, machine learning is increasingly being applied to GPR to automate interpretation. Serving as an alternative to traditional signal- and image-based methods, machine learning techniques are generally applied to post-processed GPR data and have been demonstrated to be efective in various civil engineering applications (Rasol et al. 2022).

A limited number of studies have explored the analysis of A-scan GPR data using machine learning models. In He et al. (2017), A-scan GPR data collected from tunnels in diferent regions were transformed into time-frequency maps, which were then classified using a Convolutional Neural Network (CNNs) model. The CNN approach outperformed a Support Vector Machine (SVM) benchmark in terms of accuracy. Compared to other types of GPR data, interpreting A-scan data independently is less informative and often less efective for detailed analysis.

In contrast, a larger body of research has focused on interpreting B- and C-scans. These data are often treated as raster images, making deep learning approaches, particularly CNNs, an intuitive choice, given the fact that CNNs have proven to be powerful tools for tasks such as image classification and object detection.

Some researchers use deep learning models to detect underground utilities in B-scan data (Su et al. 2023). Moreover, Tong et al. (2017) has employed CNNs to identify concealed pavement cracks from B-scan GPR data, using a dataset labeled with information derived from pavement core sampling. In subsequent studies, additional CNN-based variations (Tong et al. 2018) and networkin-network architectures (Tong et al. 2020) are applied to datasets annotated with additional classes such as subgrade unevenness and subgrade sinkholes. Several popular CNN architectures have also been adopted in this field. For instance, the AlexNet architecture (Krizhevsky et al. 2012) has been applied to pre-processed B-scan data for the detection of underground objects in urban areas (Kim et al. 2020). Various You Only Look Once (YOLO)-based frameworks (Redmon 2016) have also been employed to detect underground utilities such as cables and pipes (Zong et al. 2019), as well as concealed cracks and moisture damage in asphalt pavements (Li et al. 2021; Zhang et al. 2020). Furthermore, faster Region-based CNN frameworks have been explored for identifying buried objects by detecting hyperbolic reflections (Pham and Lefèvre 2018) and pavement distress such as cracks, water-damaged pits, and uneven settlements (Gao et al. 2020).

Beyond CNNs, other architectures such as recurrent neural networks (RNNs) have been adapted for B-scan data processing. Lei et al. (2020) integrated CNNs with Long Short-Term Memory (LSTM) networks, a type of RNN, to detect hyperbolic region features and estimate the diameters of buried cylindrical objects. In (Liu et al. 2024), a network based on the vision transformer architecture is proposed to classify pavement distress severity.

Although C-scan data provide more comprehensive and informative representations, fewer studies have focused on C-scan data. In Zhou et al. (2024), the authors propose a network architecture integrating U-Net, residual networks, and attention mechanisms to segment 3D Cscan data and reconstruct underground pipes. Other studies (Namgyu Kim and Lee 2021; Kang et al. 2019) utilize collections of B-scan and C-scan images saved as 2D grid images. These are then classified using CNN-based networks into categories such as cavities, pipes, and manholes. Similarly, in Liu et al. (2023b), one B-scan and one C-scan image are stitched together and input into a YOLO-based network to identify cracks in pavements.

## Research gaps

The above review shows that existing GPR interpretation methods have progressed from traditional signal- and image-based analysis to learning-based approaches. However, several challenges remain. Firstly, learning-based approaches mostly require annotated datasets. Currently, real-world annotated GPR datasets for pavement assessment are still limited, and the annotation process relies on expert interpretation or physical validation, both of which are dificult to scale across operational road networks. Secondly, from an algorithmic perspective, many learning-based methods focus on A-scan or B-scan data, or simplify 3D C-scan data into 2D representations. Such simplification may limit the ability of models to learn 3D volumetric information from the data.

To address the data gap, this paper proposes an RGB-guided annotation pipeline that transfers labels from pavement images to spatially corresponding 3D GPR segments. To address the algorithmic gap, this paper develops a specialised 3D convolutional neural network incorporating residual connections, mixed convolutional kernels, and depth-wise and channel-wise attention mechanisms for learning discriminative features from real-world 3D GPR data.

## METHODOLOGY

This section outlines the proposed pipeline, which includes generating a 3D GPR dataset through RGB-guided annotation and designing a deep learning architecture to interpret the annotated GPR data.

## Dataset preparation

The GPR data used in this study is part of the CAMHighways Dataset (d’Avigneau et al. 2025), collected from various sections of the UK highway network. This dataset includes point cloud data, RGB images, thermal images, and GPR data. The mobile mapping data were acquired in 2021 using a Trimble MX9 mobile mapping system, while the GPR data were collected later with a Hexagon Stream Up GPR system. Both data acquisition systems are illustrated in Figure 1. Orthomosaics, showing a top-down view of pavements generated from the pavement images after removing camera corrections, defect mask orthomosaics generated from annotations of pavement images, and GPR data are all georeferenced to the British National Grid (EPSG:27700 or OSGB36). Figure 2 shows example orthomosaics alongside their corresponding defect masks from the CAMHighways dataset.

The present GPR dataset was acquired by the above-mentioned GPR system at a diferent time from the image data collected by the mobile scanning system, resulting in diferences in the routes driven by the survey vehicle. The pavement images were captured lane by lane, with a single drive covering the imagery for one lane. However, the GPR system, with a coverage width of 1.58 m (IDS GeoRadar ), could not cover an entire lane in a single pass, requiring multiple drives to scan one lane fully. The coverage for both data sources is illustrated in Figure 3. While Figure 3a shows an example orthomosaic from the dataset, Figure 3b depicts the corresponding GPR data coverage for the same area. Notably, Figure 3c shows the overlap between the GPR coverage and orthomosaics.

In this paper, a section of concrete pavement on A14 in the UK, from Woolpit to Haugley, is used to build the image and GPR dataset. The collected GPR data is organized into 20 layers depthwise: from the ground surface to a depth of 95 cm below ground level (in increments of 5 cm). Exemplar GPR data collected across two lanes of a road section is shown in Figure 4. Figure 4a presents the 2D GPR data corresponding to the road surface $\left( \mathsf { z } = 0 \mathrm { c m } \right)$ , while Figure 4b illustrates the 2D data at a depth of z = -95 cm. Figure 4c depicts how these layers are structured in 3D, with some intermediate layers omitted for clarity.

It should also be noted that the GPR data used in this study are processed C-scan volumes exported from the manufacturer’s acquisition and processing pipeline, rather than raw time-domain A-scan traces. The manufacturer’s processing pipeline is understood to perform standard GPR preprocessing operations, such as time-zero correction, de-wow filtering, background or directwave removal, gain compensation, and band-pass filtering, before generating the C-scan slices. The exact numerical parameters of these proprietary preprocessing steps were not available to the authors.

In addition, the vertical coordinate of raw GPR data is originally represented by two-way travel time rather than physical depth. In the dataset used in this study, the time-to-depth conversion was performed internally by the manufacturer’s proprietary processing pipeline, rather than through field calibration by the authors. The specific efective dielectric constant used in this conversion was not available to the authors and is not reported in the publicly available CAMHighways documentation. Because the survey was conducted on a section of highway under continuous operation, on-site coring for dedicated wave-velocity calibration was not feasible. Therefore, the reported depths should be interpreted as nominal depth values, and their absolute accuracy may vary depending on local variations in moisture content and material composition. This does not afect the main methodological contribution of this study, because the proposed framework focuses on RGB-guided annotation and learning from co-registered GPR segments rather than estimating the absolute depth of subsurface objects.

The dataset preparation involves computing the overlapping areas between the orthomosaics and GPR data, followed by separating the areas into individual lane clusters based on their Euclidean distance. It should be noted that this method relies on the assumption that GPR data corresponding to individual lanes are always spatially separated, which is the case in the CamHighway dataset. Figure 5 illustrates example orthomosaics and GPR data collected over two lanes, along with the extracted overlapping area. Given that both RGB pavement images and GPR data were collected and organized as the survey vehicle passed the road lane by lane, the RGB-GPR dataset was constructed on a per-lane basis. This approach treats the data for each lane, along the direction of the roadway, as a separate and independent unit for subsequent processing. Figure 6 shows an example of the extracted orthomosaics and GPR data layers for a single lane.

For each lane, the next step involves generating data segments by cutting the lane’s data orthogonal to its centerline direction. This process begins by fitting a centerline curve along the lane’s longitudinal direction. Since the lane is not always perfectly straight, the centerline is modeled using a third-order polynomial, expressed as:

$$
y = \sum _ { k = 0 } ^ { 3 } p _ { k } x ^ { k }\tag{1}
$$

where � and � are the coordinates of points within the overlapping area of the orthomosaic and GPR coverage on the 2D ground plane, and $p _ { k }$ are the coeficients of the polynomial. An example of a lane’s overlapping area and the corresponding fitted centerline is shown in Figure 7.

Subsequently, key points are selected along the fitted centre line at a predefined interval, denoted as $\delta ,$ representing the length of each segment along the lane direction. In this study, � is set to 0.3 meters, which ofers a practical balance between spatial resolution and computational eficiency. Each key point is denoted as $P _ { i } = \left( x _ { i } , y _ { i } \right)$ , representing the coordinates of the $i ^ { t h }$ point along the fitted centreline. The corresponding components of the normal vector to the point of the segment are denoted $n _ { i } = ( n _ { x _ { i } } , n _ { y _ { i } } )$ . These key points define the centres of the final data segments along the lane direction. All points within the overlapped area are assigned to segments based on their nearest normal projection. The distance from a point $\boldsymbol { Q _ { p } } = ( x _ { p } , y _ { p } )$ in the overlapped area to the $i ^ { t h }$ segment can be calculated as:

$$
d _ { p , i } = \mid ( x _ { p } - x _ { i } ) \cdot n _ { y _ { i } } - ( y _ { p } - y _ { i } ) \cdot n _ { x _ { i } } \mid .\tag{2}
$$

Each point $Q _ { p }$ is assigned to the segment whose centreline point is closest to it, based on the minimum distance $d _ { p , i }$ . The index of the nearest segment is determined as

$$
i ^ { * } = \operatorname * { a r g m i n } _ { i } d _ { p , i } .\tag{3}
$$

Through this process, each point $Q _ { p }$ in the overlapped area is assigned to a segment. The final dataset can thus be viewed as the result of dividing the lane into a series of segments using a moving plane orthogonal to the lane’s centreline along its length, which is illustrated in Figure 8, where diferent colours distinguish adjacent segments. The corresponding pseudocode is provided in Algorithm 1. Each segment has a size of $1 0 0 \times 2 5 \times 2 0 \times 3$ , where 100 represents the length along the lane (longitudinal direction), 25 represents the lateral direction across the lane, 20 is the vertical depth, and 3 denotes the RGB color channels.

Algorithm 1 Assign GPR Points to Nearest Segment   
Inputs:   
Key Points: List of key points along the fitted central curve, $P _ { 1 } , \ldots , P _ { n } .$ , where each $P _ { i } =$   
$( x _ { i } , y _ { i } )$   
Normal Vectors: List of normal vectors at each key point, $\mathbf { n } _ { 1 } , \ldots , \mathbf { n } _ { n } .$ , where each $\mathbf { n } _ { i } ~ =$   
$( n _ { x _ { i } } , n _ { y _ { i } } )$ is the normal vector at $P _ { i }$   
GPR Points: List of GPR points to be assigned, $Q _ { 1 } , \ldots , Q _ { m } .$ , where each $Q _ { p } = ( x _ { p } , y _ { p } )$ is a   
point to be assigned   
Outputs:   
Segment Indices: List of indices $i ^ { * }$ representing the closest segment for each GPR point   
Initialization:   
Create an empty list segment\_indices to store the index $i ^ { * }$ of the closest segment for each GPR   
point.   
for each GPR Point $Q _ { p } = ( x _ { p } , y _ { p } )$ in the list of GPR Points do   
Set min\_distance\_Q\_p to a very large value (∞).   
Set best\_segment\_index\_ $\mathsf { \Omega } _ { - } \mathsf { p } \tan - 1$   
for each Segment � (Defined by $P _ { i } = ( x _ { i } , y _ { i } )$ and $n _ { i } = ( n _ { x _ { i } } , n _ { y _ { i } } ) )$ do   
Compute the distance $d _ { p , i }$ from $Q _ { p }$ to segment � using the formula:   
$d _ { p , i } = \left| ( x _ { p } - x _ { i } ) \cdot n _ { y _ { i } } - ( y _ { p } - y _ { i } ) \cdot n _ { x _ { i } } \right|$   
if $d _ { p , i }$ is less than min\_distance then   
Set min\_distance $\mathsf { \Omega } _ { \mathsf { - } } \mathsf { P }$ to $d _ { p , i }$   
Set best\_segment\_index $\mathsf { \Omega } _ { - } \mathsf { Q } _ { - } \mathsf { p }$ to �.   
end if   
end for   
Append best\_segment\_index\_Q\_p to segment\_indices.   
end for   
Output:   
The list segment\_indices contains the index �<sup>∗</sup> of the nearest segment for each GPR point   
$Q _ { p }$  
Figure 9 presents visualisations of example data segments in the final constructed dataset. Figure 9a, 9b, and 9c show segments containing a patch defect, a crack defect, and no defect, respectively, as seen on the orthomosaics. Figure 9d illustrates the orthomosaic of a segment alongside its corresponding GPR data layers, displayed layer by layer.

## Network design

A 3D CNN with residual connections and attention mechanisms is designed for the classification of 3D GPR segments. This network uses a combination of advanced building blocks, including mixed-kernel residual modules, and depthwise feature attention mechanisms, to enhance feature representation and learning eficiency. The designed architecture is shown in Figure 10.

## Residual block with mixed kernel convolutions and channel-wisefeature attention

This module is a custom 3D residual block designed to enhance feature representation by combining mixed kernel sizes of convolutional computing and Squeeze-and-excitation (SE) attention (Hu et al. 2018). It has two parallel convolutional paths to extract features with diferent receptive fields and leverages an SE block to recalibrate channel-wise feature responses. A residual connection is employed to preserve input information and facilitate gradient flow during training.

The input tensor to the block is denoted $X ~ \in ~ \mathbb { R } ^ { B \times C _ { \mathrm { i n } } \times D \times H \times W }$ , where $B , C _ { \mathrm { i n } } , D , H$ , and � represent the batch size, number of input channels, depth, height, and width, respectively. The block processes the input using two parallel paths.

The first path applies a $3 \times 3 \times 3$ convolution with kernel $W _ { 1 } \in \mathbb { R } ^ { ( C _ { \mathrm { o u t } } / 2 ) \times C _ { \mathrm { i n } } \times 3 \times 3 \times 3 }$ , with stride �, and padding $p = 1$ , followed by a batch normalization layer $\mathrm { B N } _ { 1 } ( \cdot )$ and a ReLU activation function:

$$
\mathrm { P a t h } _ { 1 } ( X ) = \mathrm { R e L U } \big ( \mathbf { B } \mathbf { N } _ { 1 } ( W _ { 1 } * X + b _ { 1 } ) \big ) ,\tag{4}
$$

where $b _ { 1 } \ \in \ \mathbb { R } ^ { C _ { \mathrm { o u t } } / 2 }$ is the bias term, $C _ { \mathrm { o u t } } / 2$ denotes the number of output channels (or filters) produced by the convolution layer, and ∗ denotes the convolution operation. This produces an output tensor of dimensions $\mathrm { P a t h } _ { 1 } ( X ) \ \in \ \mathbb { R } ^ { B \times ( C _ { \mathrm { o u t } } / 2 ) \times D ^ { \prime } \times H ^ { \prime } \times W ^ { \prime } }$ , where $D ^ { \prime } , H ^ { \prime } , W ^ { \prime }$ are the spatial dimensions of the output result.

The second path applies a mixed kernel convolution using a $1 \times 3 \times 3$ kernel. The operation uses kernel $W _ { 2 } \in \mathbb { R } ^ { ( C _ { \mathrm { o u t } } / 2 ) \times C _ { \mathrm { i n } } \times 1 \times 3 \times 3 }$ , followed by a batch normalization layer $\mathrm { B N } _ { 2 } ( \cdot )$ and a ReLU activation function:

$$
\mathrm { P a t h } _ { 2 } ( X ) = \mathrm { R e L U } \bigl ( \mathbf { B } \mathbf { N } _ { 2 } ( W _ { 2 } * X + b _ { 2 } ) \bigr ) ,\tag{5}
$$

where the bias term $b _ { 2 } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } / 2 }$ . The output tensor from this path is $\mathrm { P a t h } _ { 2 } ( X ) \in \mathbb { R } ^ { B \times ( C _ { \mathrm { o u t } } / 2 ) \times D ^ { \prime } \times H ^ { \prime } \times W ^ { \prime } }$

The outputs of the two paths are concatenated together along the channel dimension and fused through a $3 \times 3 \times 3$ convolution with kernel $W _ { \mathrm { m e r g e } } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times C _ { \mathrm { o u t } } \times 3 \times 3 \times 3 }$ , and a batch normalization layer $\mathrm { B N } _ { \mathrm { m e r g e } } ( \cdot )$ :

$$
Z = \mathrm { B N } _ { \mathrm { m e r g e } } \left( W _ { \mathrm { m e r g e } } * \mathrm { C o n c a t } ( \mathrm { P a t h } _ { 1 } ( X ) , \mathrm { P a t h } _ { 2 } ( X ) ) + b _ { \mathrm { m e r g e } } \right) ,\tag{6}
$$

where $b _ { \mathrm { m e r g e } } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } }$ is the bias term, Concat(·) denotes the concatenation operation.

After fusing the features, the SE block (Hu et al. 2018) is integrated to perform channelwise attention. The SE block is a neural network module designed to enhance a network’s representational capacity by modelling interdependencies between the channels of convolutional features. This block is incorporated into the present network to re-weight the feature maps of each channel during the processing of GPR segments. By applying attention mechanisms to channel dimensions, the SE block enables feature recalibration and allows the network to selectively emphasize informative features.

In the SE block, first, global average pooling is used to aggregate spatial information for each channel of the feature map tensor $Z \in \mathbb { R } ^ { B \times C _ { \mathrm { o u t } } \times D ^ { \prime } \times H ^ { \prime } \times W ^ { \prime } }$ , where � is the batch size, $C _ { \mathrm { o u t } }$ is the number of output channels, and $D ^ { \prime } , H ^ { \prime } , W ^ { \prime }$ denote the depth, height, and width of the feature maps, respectively. The pooled channel-wise descriptors are computed as:

$$
S _ { c } = \frac { 1 } { D ^ { \prime } H ^ { \prime } W ^ { \prime } } \sum _ { d = 1 } ^ { D ^ { \prime } } \sum _ { h = 1 } ^ { H ^ { \prime } } \sum _ { w = 1 } ^ { W ^ { \prime } } Z _ { b c d h w } ,\tag{7}
$$

producing a channel descriptor $S _ { c } \in \mathbb { R } ^ { B \times C _ { \mathrm { o u t } } }$ . Two fully connected layers with a ReLU activation and a Sigmoid function are then applied to compute the channel-wise attention weights:

$$
\alpha _ { c } = \sigma \big ( W _ { 2 } ^ { \mathrm { S E } } \mathrm { R e L U } ( W _ { 1 } ^ { \mathrm { S E } } S + b _ { 1 } ^ { \mathrm { S E } } ) + b _ { 2 } ^ { \mathrm { S E } } \big ) ,\tag{8}
$$

where $W _ { 1 } ^ { \mathrm { S E } } ~ \in ~ \mathbb { R } ^ { r \times C _ { \mathrm { o u t } } }$ and $W _ { 2 } ^ { \mathrm { S E } } \ \in \ \mathbb { R } ^ { C _ { \mathrm { o u t } } \times r }$ are the weight matrices of the two FC layers, and

$b _ { 1 } ^ { \mathrm { S E } } \in \mathbb { R } ^ { r } , b _ { 2 } ^ { \mathrm { S E } } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } }$ are the corresponding bias terms, and � is the reduction ratio. The function $\sigma ( \cdot )$ denotes the Sigmoid activation. Finally, the learned attention weights $\alpha _ { c }$ are applied to recalibrate the original feature maps through channel-wise multiplication:

$$
Z _ { c } ^ { \mathrm { S E } } = \alpha _ { c } \cdot Z _ { c } ,\tag{9}
$$

where $Z _ { c } ^ { \mathrm { S E } }$ is the recalibrated feature map for the �-th channel of feature map $Z _ { c }$

A shortcut connection is then added to preserve the input information. $\mathrm { ~ A ~ } 1 \times 1 \times 1$ convolution is used to match the dimensions of features:

$$
\mathrm { S h o r t c u t } ( X ) = \mathrm { B N } _ { \mathrm { s h o r t c u t } } \big ( W _ { \mathrm { s h o r t c u t } } * X + b _ { \mathrm { s h o r t c u t } } \big ) ,\tag{10}
$$

where $\mathrm { B N } _ { \mathrm { m e r g e } } ( \cdot )$ is the batch normalization layer, $W _ { \mathrm { s h o r t c u t } } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } } \times 1 \times 1 \times 1 }$ and $b _ { \mathrm { s h o r t c u t } } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } }$ are the weight tensor of the $1 \times 1 \times 1$ convolution and the corresponding bias term. The final output is computed as:

$$
Y = { \mathrm { R e L U } } ( Z _ { c } ^ { \mathrm { S E } } + { \mathrm { S h o r t c u t } } ( X ) ) .\tag{11}
$$

The output tensor $Y \in \mathbb { R } ^ { B \times C _ { \mathrm { { o u t } } } \times D ^ { \prime } \times H ^ { \prime } \times W ^ { \prime } }$ represents the recalibrated feature maps with enhanced representational capacity.

## Depthwise feature attention module

The depthwise feature attention module is designed to selectively emphasize informative features along the depth dimension of a 3D input tensor. This attention mechanism operates independently for each depth segment and is implemented using a sequence of neural network layers.

Let the input tensor be $X \in \mathbb { R } ^ { B \times C \times D \times H \times W }$ , following the same indexing as above where �, �, �, �, and � represent the batch size, number of channels, depth, height, and width, respectively. The module first applies an adaptive average pooling operation across the spatial dimensions � and

�, retaining the depth dimension �:

$$
Y = \frac { 1 } { H W } \sum _ { h = 1 } ^ { H } \sum _ { w = 1 } ^ { W } X _ { b c d h w }\tag{12}
$$

resulting in a tensor � of dimensions $B \times C \times D \times 1 \times 1$

Subsequently, two $1 \times 1 \times 1$ convolutional layers process �. The first convolution applies a transformation with kernel $W _ { 1 } ^ { \mathrm { D A } } \in \mathbb { R } ^ { C \times C \times 1 \times 1 \times 1 }$ and bias $b _ { 1 } ^ { \mathrm { D A } } \in \mathbb { R } ^ { C }$ , followed by a ReLU activation:

$$
Z ^ { ( 1 ) } = \mathrm { R e L U } ( W _ { 1 } ^ { \mathrm { D A } } * Y + b _ { 1 } ^ { \mathrm { D A } } ) .\tag{13}
$$

The second convolution applies a transformation with kernel $W _ { 2 }$ and bias $b _ { 2 }$ , followed by a sigmoid activation, reducing the channel dimension to 1:

$$
Z ^ { ( 2 ) } = \mathrm { S i g m o i d } ( W _ { 2 } ^ { \mathrm { D A } } * Z ^ { ( 1 ) } + b _ { 2 } ^ { \mathrm { D A } } ) ,\tag{14}
$$

where $W _ { 2 } ^ { \mathrm { D A } } \in \mathbb { R } ^ { 1 \times C \times 1 \times 1 \times 1 }$ is the weight matrix and $b _ { 2 } ^ { \mathrm { D A } } \in \mathbb { R } ^ { 1 }$ is the bias term, producing output $Z ^ { ( 2 ) }$ of dimensions $B \times 1 \times D \times 1 \times 1$

These scores are normalized using a softmax function across the depth dimension �:

$$
A _ { b d } = \frac { e ^ { Z _ { b d } ^ { ( 2 ) } } } { \sum _ { k = 1 } ^ { D } e ^ { Z _ { b k } ^ { ( 2 ) } } } ,\tag{15}
$$

where $( A _ { b d } \in \mathbb { R } ^ { B \times D }$ denotes the attention scores, with one score per depth segment � for each sample in the batch �. This ensures that each depth segment receives a weighted emphasis based on its computed significance.

Finally, the attention scores are expanded back to the dimensions of the input tensor and applied element-wise:

$$
O _ { b c d h w } = A _ { b d } \times X _ { b c d h w } ,\tag{16}
$$

where $O \in \mathbb { R } ^ { B \times C \times D \times H \times W }$ is the output tensor $X \in \mathbb { R } ^ { B \times C \times D \times H \times W }$ , which retains the same dimen-

sions as the input � but includes features along the depth dimension according to their learned significance. By reweighting the input tensor, this module emphasizes the most informative regions across the depth segments.

## EXPERIMENTS AND RESULTS

## Implementation details

The approach of constructing the GPR-RGB dataset is detailed in Section 4. Two types of pavement defects in the original CamHighway dataset are extracted on the pavement images: patches and cracks. These annotations were then transferred to the corresponding GPR-RGB data segments. Two examples of GPR segments with patch and crack defects are illustrated in Figure 11. Specifically, if a defect is visible in the RGB image, the associated GPR segment is labelled with the same defect type. These labels are treated as RGB-guided proxy labels for learning correlations between surface-visible defects and GPR response patterns, rather than as independently verified ground truth for subsurface distress. Therefore, the proposed annotation strategy is applicable mainly to defects with visible surface manifestations and does not provide labels for hidden internal distresses that are not observable from images. No additional field spotchecks, coring, or independent physical validation of the model predictions were conducted, as the data were collected from a highway section under continuous operation. The total number of generated segments, along with the sample distribution across the two defect classes and the train-test split, is summarised in Table 1.

The dataset preparation and the neural networks were implemented in Python. The experiments were conducted on the customized dataset described above on an NVIDIA A100 GPU with 80 GB of memory.

## Results and outcome analysis

To ensure a fair performance comparison, each experimental model is trained and tested four times, and the average of the resulting evaluation metrics is reported. All tests are formulated as binary classification problems. Specifically, these models are trained on data labelled as "patch vs. normal" and "crack vs. normal". To evaluate the performance of the proposed network, three neural networks are implemented: a) ResNet-18 (He et al. 2016) with and without pre-training on the 2D ImageNet dataset (Deng et al. 2009). Given the significant diferences between natural images and GPR data, we hypothesise that pre-training may not yield noticeable performance improvements. b) A 3D CNN with the same number of layers and convolutional kernel sizes as the proposed architecture, but without additional enhancements. c) The proposed network with residual blocks, mixed convolutional kernel sizes, and both channel-wise and depth-wise feature attention mechanisms. To ensure a fair evaluation, all networks have the same configuration: a batch size of 2, a maximum of 50 epochs. The AdamW optimiser is employed, which improves generalisation by using weight decay from gradient updates. A step learning rate scheduler was applied with a step size of 10 and a decay factor $\gamma$ of 0.1. To address class imbalance in the training data, a weighted binary cross-entropy loss function $\mathcal { L }$ was used to assign higher penalties to underrepresented classes during model training:

$$
\mathcal { L } = - \left( w _ { 1 } y \log ( p ) + w _ { 0 } ( 1 - y ) \log ( 1 - p ) \right) ,\tag{17}
$$

where $y \in \{ 0 , 1 \}$ is the ground truth label, $p \in [ 0 , 1 ]$ is the predicted probability for the positive class, and $w _ { 1 } , w _ { 0 } \in \mathbb { R } ^ { + }$ are the weights for the positive and negative classes, respectively. Figure 12 shows the training accuracy curves of the proposed network for both the patch and crack classification tasks.

We use the following standard metrics to evaluate the performance of the networks: accuracy, precision, recall, and F1 score. These metrics are defined by the following equations:

$$
\mathrm { A c c u r a c y } = { \frac { \mathrm { T P } + \mathrm { T N } } { \mathrm { T P } + \mathrm { T N } + \mathrm { F P } + \mathrm { F N } } } ,\tag{18}
$$

$$
\mathrm { P r e c i s i o n } = \frac { \mathrm { T P } } { \mathrm { F P } + \mathrm { T P } } ,\tag{19}
$$

$$
{ \mathrm { R e c a l l } } = { \frac { \mathrm { T P } } { \mathrm { T P } + { \mathrm { F N } } } } ,\tag{20}
$$

$$
{ \mathrm { F 1 ~ S c o r e } } = { \frac { 2 \times { \mathrm { P r e c i s i o n } } \times { \mathrm { R e c a l l } } } { { \mathrm { P r e c i s i o n } } + { \mathrm { R e c a l l } } } } ,\tag{21}
$$

where TP, TN, FP, and FN denote true positive, true negative, false positive, and false negative, respectively. The quantitative evaluation of diferent networks on the test set is presented in Table 2. The reported values for precision, recall, and F1-score are macro-averaged across two classes, which averages these metrics for the two classes to assess the overall performance, providing a comprehensive assessment of each model’s efectiveness.

For patch defect detection, the ResNet-18 network achieved accuracies of 59.6% and 59.3% for models without and with pre-training on the ImageNet dataset (Deng et al. 2009), respectively. Each GPR segment is represented as a tensor of size $1 0 0 \times 2 5 \times 2 0 \times 3$ , where the dimensions correspond to the longitudinal direction, lateral direction, depth, and RGB colour channels, respectively. Since the original ResNet-18 model is designed to process standard 2D images with three channels, the GPR segments were reformatted to $1 0 0 \times 2 5 \times 6 0$ by flattening the depth and colour dimensions into a single channel dimension, whilst preserving the spatial resolution. To support this input format, the first convolutional layer of the ResNet-18 architecture was modified to accommodate the transformed GPR segments. As shown in Table 2, pre-training with weights derived from conventional 2D image datasets does not improve performance metrics. Although transfer learning typically enhances object detection accuracy in diferent applications in the Architecture, Engineering, and Construction industry (Pan et al. 2022a; Pan et al. 2022b), the features learned from purely 2D images do not efectively transfer to 3D GPR segment data. This limitation likely arises from the fundamental diferences in dimensionality and data characteristics between 2D imagery and 3D GPR representations.

For the patch classification task, compared to the ResNet-18 model, which processes GPR data as 2D data, the 3D-CNN model provides significantly improved performance, increasing the accuracy from 59.6% to 67.6%, precision from 58.9% to 66.9%, recall from 59.0% to 65.8%, and F1 score from 58.9% to 65.9%. This improvement is expected since the 3D convolutional layers are adept at extracting features from 3D spaces. Given that GPR segments inherently possess

a 3D structure, the ability to extract features along the depth dimension is crucial for accurately predicting the classes of these segments.

Building on the enhancements provided by the 3D convolutional layers, the proposed network incorporates residual connections, mixed convolutional kernel sizes, and attention mechanisms, resulting in further improvements across the various performance metrics. Specifically, accuracy increases from 67.6% to 69.0%, precision from 66.9% to 68.2%, recall from 65.8% to 68.0%, and F1-score from 65.9% to 68.1%. These results demonstrate that the proposed model provides meaningful and consistent improvements over existing baselines. Its efectiveness is particularly notable given the challenges posed by real-world GPR data, which are often noisy and ambiguous due to the complexity of road environments.

For crack defect detection, similar patterns in prediction results were observed. The ResNet-18 models achieved lower accuracies of 54.0% and 54.9% with and without pre-training on ImageNet, respectively. Once again, the 3D-CNN model demonstrated a notable improvement, increasing the accuracy to 69.1%. However, the present network provides slightly improved performance, achieving the highest predicted accuracy of 69.6%.

The performance improvement from the 3D CNN to the proposed network is modest compared to the gain from ResNet-18 to the 3D CNN. To assess its significance, we conducted paired t-tests across four training runs using identical data splits. This test is suitable here as it evaluates diferences under matched conditions. For patch classification, the p-value is 0.0272, indicating a statistically significant improvement ( $p \ < \ 0 . 0 5 )$ at the 95% confidence level. For crack classification, the p-value is 0.0938, suggesting moderate evidence of improved performance at the 90% confidence level, though not significant at 95%. In summary, the proposed network significantly outperforms the 3D CNN in patch classification and shows a consistent, albeit weaker, advantage in crack classification.

Despite achieving similar overall accuracies of approximately 70%, the model’s performance for two tasks, patch defect detection and crack defect detection, varies when examining the patch and crack classes in detail. This variation is detailed in Table 3, where class-wise precisions, recalls,

and F1 scores are presented. It is evident that the model’s ability to detect patches is substantially better than identifying cracks, a finding that is supported by the confusion matrices presented in Figure 13.

This discrepancy can be attributed to the following factors: a) The causal link between GPR anomalies and surface defects is likely to difer between patches and cracks. For instance, when the GPR anomaly reflects increased subsurface water content, the presence of a patch is more likely to correlate strongly. Patches often result from prior surface damage that permits significant water ingress into the underlying material. In contrast, cracks are typically narrow and allow limited water penetration, leading to subtler subsurface changes that produce weaker GPR responses. b) The model’s ability to detect patches and cracks is diferent. Patch defects are generally larger than cracks, making them more detectable with the employed techniques. Furthermore, patch defects tend to have more regular shapes as they are human-generated during repair processes, whereas crack defects occur more randomly and result from various causes, presenting diverse manifestations across environments. c) The spatial resolution of the GPR data in this study is approximately 5 cm, which leads to information loss for finer-scale features. This limitation particularly afects the detection of cracks, which are often narrow and fall below the resolution threshold. d) The data utilized in this study are collected from real-world environments and are highly noisy. This background noise likely obscures subtle patterns associated with cracks, making them harder to distinguish from the surrounding signal.

It should also be noted that the classification labels used in this study are derived from surfacevisible image defects. Therefore, the reported results evaluate the ability of the model to recognise GPR patterns associated with these visible defects, rather than its ability to identify all forms of subsurface distress. In particular, defects such as voids or delamination that do not show surface manifestations are outside the scope of the present dataset and model.

The experiments in this study were formulated as two binary classification tasks, namely patch versus normal and crack versus normal, because the primary objective was to evaluate whether GPR segments associated with each type of surface-visible defect could be distinguished from normal pavement segments. This formulation is also consistent with preliminary screening workflows in road inspection, where candidate abnormal segments are first identified for further review. A three-class setting involving normal, patch, and crack classes was also considered. However, preliminary results indicated that the current model and dataset have dificulty distinguishing crack and patch segments from each other. This suggests that the GPR response patterns associated with RGB-labelled cracks and patches are not always clearly separable in the current dataset, likely due to the weak-label nature of RGB-guided annotation, limited GPR resolution, and noisy real-world highway data.

From the perspective of practical road maintenance, the proposed model should be regarded as a preliminary screening or decision-support tool rather than a standalone system for determining maintenance actions. The relatively low F1-score for crack detection and the moderate recall for patch detection indicate that false positives and false negatives remain substantial. False positives may increase unnecessary follow-up inspections, while false negatives may cause relevant defects to be missed, which is more critical in safety-related scenarios. Therefore, the model outputs should be used to select candidate road segments for further inspection, rather than directly starting maintenance actions. Further improvements in data quality, dataset size, model robustness, and independent field validation are needed before operational deployment.

## Ablation study

The network architecture proposed in Section 4 incorporates 3D convolutional layers enhanced with residual connections, and utilizes mixed convolutional kernel sizes along with depthwise and channelwise attention mechanisms to optimize feature extraction from GPR segments. Expanding on the standard 3D CNN framework, various configurations are explored: networks with individual integrations of residual connections, depth attention, channel attention, and mixed kernel sizes, as well as the proposed network that combines all these techniques. The comparative performance of these configurations for patch classification is detailed in Table 4.

Similar to the experiments outlined in Table 2, all networks were trained and tested five times, with the average values from these trials reported in Table 4. The ablation results show that the contribution of each architectural component is not uniformly positive when evaluated independently. Compared with the baseline 3D CNN, the depth attention mechanism improves all four evaluation metrics, increasing the accuracy from 67.6% to 68.3% and the F1-score from 65.9% to 67.3%. The channel attention mechanism provides modest improvements in recall and F1-score, although its accuracy remains similar to the baseline. The residual block and mixed-kernel module do not improve accuracy when used independently.

These results suggest that the proposed architecture benefits from the synergistic interaction among its components rather than from a simple cumulative improvement of each individual module. Although residual connections and mixed kernels do not improve performance in isolation, they may support feature propagation and multi-scale feature representation when combined with depth-wise and channel-wise attention.

## Influence of GPR input layers

As described in Section 4, the GPR data utilized in this study contain 20 layers, extending from the pavement surface downward into the subsurface, with a 5 cm interval between adjacent layers. Table 5 compares the performance of the proposed network in classifying patch defects across four configurations of input layers: 5, 10, 15, and 20 layers from the top. It can be observed that configurations with a larger number of input layers achieve better scores, with the 5-layer input performing the worst and the 20-layer input demonstrating the best prediction results. However, this trend should be interpreted with caution. Patch defects usually extend only to a limited depth. Therefore, the improved classification performance obtained by including more GPR slices should not be interpreted as evidence that the physical patch defect extends to the deepest layers included in the model input.

A more reasonable interpretation is that the additional GPR slices provide contextual information that helps the model distinguish patched and non-patched segments. The upper GPR slices are expected to contain the most direct information related to patch materials, particularly when the dielectric properties of the patch material difer from those of the neighbouring materials. Deeper slices may still contribute indirectly by capturing broader GPR signal characteristics, such as reflection patterns, attenuation behaviour, or moisture-related dielectric changes. In this sense, the benefit of using more slices reflects the contribution of 3D contextual information rather than the physical depth extent of the patch defect itself.

However, no such trend is observed for crack defects within the dataset. From Table 6, utilizing only the top 5 layers results in the poorest prediction outcomes. However, adding more layers did not consistently improve all metrics. This variability can be attributed to the irregular nature of cracks compared to patches, which are more uniformly constructed during pavement maintenance. Cracks vary significantly based on diferent factors such as their causes and environmental conditions, presenting diverse properties and appearances. Deeper layers in the GPR data may decrease model performance when they are irrelevant to the defect itself. However, the low spatial resolution and high noise levels in the collected GPR data, combined with the irregular and small-scale nature of cracks, make crack detection particularly challenging. These issues also afect the ability to analyse and identify performance trends when varying the number of input GPR layers along the depth dimension.

## CONCLUSIONS

This paper presented a novel pipeline for analysing 3D GPR data using RGB-guided annotation and deep learning. We introduced a cost-efective method to construct annotated 3D GPR datasets by aligning orthomosaic images with GPR scans, and designed a specialised 3D convolutional neural network enhanced with residual connections, attention mechanisms, and mixed kernel sizes to improve feature extraction. The proposed network was evaluated on patch and crack detection tasks, with experimental results demonstrating its better performance over baseline approaches. This paper addresses the research gaps discussed and makes significant contributions to state-ofthe-art GPR research in several key aspects:

a) This paper introduced a novel network architecture specifically designed to process 3D GPR data and classify two common types of pavement defects: patches and cracks. The proposed network demonstrated meaningful and consistent improvements over existing baselines.

b) A 3D GPR dataset collected in the real world on a section of concrete pavement on a section of highway in use (A14 in the UK) was presented. It enabled the comparative evaluation of the performance of various models on real-world data. It is important to emphasize the value of a real-world dataset. The GPR data contained within are notably more complex, irregular, and noisy, due to numerous factors that cannot be modeled in a GPR simulator.

c) This paper presented a novel approach involving the alignment of RGB-GPR data, using pavement RGB images as a reference for annotating GPR data. This approach enabled the generation of a large volume of labels by first labelling the images and then transferring these annotations to the corresponding GPR segments. Given that labelling an image is significantly faster and more cost-efective compared to directly labelling GPR data, which often requires techniques like coring or specialized human expertise, this approach ofers a practical alternative for data annotation. Furthermore, the annotated RGB-GPR segments provide possible future research directions, such as exploring deep learning techniques across these two data modalities.

Practically, this study makes important contributions to the civil and transportation engineering domain by proposing a scalable and cost-efective approach for pavement defect detection using GPR data. The novel idea of using RGB images as a reference to label 3D GPR data addresses a major bottleneck in the field: the lack of real-world annotated datasets. It enables large-scale data generation under real-world conditions. With such datasets available, more advanced machine learning and deep learning models can be developed and applied to improve defect recognition performance.

Furthermore, the integration of GPR and pavement image data improves both the eficiency and depth of pavement condition assessment, allowing not only for the detection of surface and subsurface defects but also for the analysis of underlying causes. This integrated insight is especially valuable for pavement engineers and supports intelligent platforms like road digital twins (Brilakis et al. 2019; Pan et al. 2024), as it supports more informed and proactive decision-making in maintenance planning and repair prioritisation, contributing to more efective and data-driven road asset management.

However, this study still has the following limitations: a) the RGB-guided annotation strategy provides proxy labels rather than physically verified ground truth for subsurface distress. This study assumes that defects visible in RGB images always correspond to distinguishable defect patterns in GPR data. However, there is no guaranteed correlation that a defect on the pavement surface always corresponds to a subsurface anomaly, or vice versa. Accurately annotating these subsurface defects usually requires complementary approaches, such as coring or field checks. This limitation applies to the prediction results as well. The validation and testing phases were based on samples from the constructed dataset, without independent physical validation to verify the model predictions. Future work should include independent validation through coring, field spot-checks, or complementary non-destructive testing methods, where feasible, to further assess the reliability of the method in practical pavement maintenance. b) The characteristics of GPR data can vary substantially across diferent environments, depending on subsurface material properties such as density, composition, and moisture content. For example, GPR signals interact diferently with sandy soils compared to clayey soils due to diferences in dielectric properties. These variations can change the appearance of anomaly patterns in the GPR data. As a result, the model trained on data collected from a specific environment may not generalise well to GPR data acquired under diferent ground conditions or in diferent geographical locations. Models trained on the dataset used in this study might not perform efectively on GPR data collected from other environments. c) The current model performance, especially for crack recognition, is not yet suficient for standalone operational decision-making. In practical road maintenance, the model should be used as a screening tool to support further inspection. Further improvements in aspects like data quality, data resolution, and model robustness are needed before real operational deployment.

In future work, we aim to expand the GPR-RGB dataset to include a wider variety of pavement types, and additional geographical areas.A more expansive dataset will enable the development of more comprehensive models capable of performing additional tasks.Furthermore, we intend to include other types of pavement defects, such as rutting, bleeding, ravelling, and potholes, to enhance the dataset’s applicability and support more generalised defect detection.

## DATA AVAILABILITY STATEMENT

Data that support the findings of this study are available from the corresponding author upon reasonable request.

## ACKNOWLEDGEMENTS

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 101034337.

## REFERENCES

ASTM International. ASTM D6432-19 Standard Guide for Using the Surface Ground Penetrating Radar Method for Subsurface Investigation, <https://www.astm.org/d6432-19.html> (September 5, 2025).

Brilakis, I., Pan, Y., Borrmann, A., Mayer, H.-G., Rhein, F., Vos, C., Pettinato, E., and Wagner, S. (2019). Built Environment Digital Twining, <https://www.repository.cam.ac.uk/handle/1810/318329>.

Bruschini, C., Gros, B., Guerne, F., Pièce, P.-Y., and Carmona, O. (1998). “Ground penetrating radar and imaging metal detector for antipersonnel mine detection.” Journal of Applied Geophysics, 40(1-3), 59–71.

Chang, C. W., Lin, C. H., and Lien, H. S. (2009). “Measurement radius of reinforcing steel bar in concrete using digital image gpr.” Construction and Building Materials, 23(2), 1057–1063.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. (2009). “Imagenet: A large-scale hierarchical image database.” 2009 IEEE conference on computer vision and pattern recognition, Ieee, 248–255.

Department for Transport, UK. Road Trafc Estimates: Great Britain 2021, <https://assets.publishing.service.gov.uk/media/633310c1d3bf7f567b3e61fb/road-traficestimates-in-great-britain-2021.pdf> (September 5, 2025).

Department for Transport, UK. Transport Statistics Great Britain 2019, <https://assets.publishing.service.gov.uk/media/5e610a50d3bf7f108502ecaa/tsgb-2019.pdf> (September 5, 2025).

Department for Transport, UK. Transport Statistics Great Britain: 2022 Domestic Travel, <https://www.gov.uk/government/statistics/transport-statistics-great-britain-2022/transportstatistics-great-britain-2022-domestic-travel> (September 5, 2025).

Dou, Q., Wei, L., Magee, D. R., and Cohn, A. G. (2017). “Real-time hyperbola recognition and fitting in gpr data.” IEEE Transactions on Geoscience and Remote Sensing, 55(1), 51–62.

d’Avigneau, A. M., Potseluyko, L., Anvo, N. R., Taha, H. M., Reja, V. K., Davletshina, D., Lam, P., de Silva, L., Al-Tabbaa, A., and Brilakis, I. (2025). “Camhighways: The cambridge highways dataset.” Advanced Engineering Informatics, 64, 103036.

Frigui, H., Zhang, L., and Gader, P. D. (2010). “Context-dependent multisensor fusion and its application to land mine detection.” IEEE Transactions on Geoscience and Remote Sensing, 48(6), 2528–2543.

Gao, J., Yuan, D., Tong, Z., Yang, J., and Yu, D. (2020). “Autonomous pavement distress detection using ground penetrating radar and region-based deep learning.” Measurement, 164, 108077.

He, K., Zhang, X., Ren, S., and Sun, J. (2016). “Deep residual learning for image recognition.” Proceedings ofthe IEEE conference on computer vision and pattern recognition, 770–778.

He, Y.-y., Li, B.-q., Guo, Y.-s., Wang, T.-n., and Zhu, Y. (2017). “An interpretation model of gpr point data in tunnel geological prediction.” Eighth International Conference on Graphic and Image Processing (ICGIP 2016), Vol. 10225, SPIE, 520–525.

Hu, J., Shen, L., and Sun, G. (2018). “Squeeze-and-excitation networks.” Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June).

IDS GeoRadar. Ground Penetrating Radar System: Stream UP, <https://idsgeoradar.com/products/ground-penetrating-radar/stream-up> (June 9, 2025).

Jing, H. and Vladimirova, T. (2017). “Novel algorithm for landmine detection using c-scan ground penetrating radar signals.” 2017 Seventh International Conference on Emerging Security Technologies (EST), 68–73.

Jol, H. M. (2008). Ground penetrating radar theory and applications. elsevier.

Kang, M.-S., Kim, N., Im, S. B., Lee, J.-J., and An, Y.-K. (2019). “3d gpr image-based ucnet for

enhancing underground cavity detectability.” Remote Sensing, 11(21).

Kim, N., Kim, K., An, Y.-K., Lee, H.-J., and Lee, J.-J. (2020). “Deep learning-based underground object detection for urban road pavement.” International Journal of Pavement Engineering, 21(13), 1638–1650.

Klęsk, P., Godziuk, A., Kapruziak, M., and Olech, B. (2015). “Fast analysis of c-scans from ground penetrating radar via 3-d haar-like features with application to landmine detection.” IEEE Transactions on Geoscience and Remote Sensing, 53(7), 3996–4009.

Krizhevsky, A., Sutskever, I., and Hinton, G. E. (2012). “Imagenet classification with deep convo lutional neural networks.” Advances in neural information processing systems, 25.

Kuchipudi, S. T., Ghosh, D., and Gupta, H. (2022). “Automated assessment of reinforced concrete elements using ground penetrating radar.” Automation in Construction, 140, 104378.

Lei, W., Luo, J., Hou, F., Xu, L., Wang, R., and Jiang, X. (2020). “Underground cylindrical objects detection and diameter identification in gpr b-scans via the cnn-lstm framework.” Electronics, 9(11).

Li, S., Gu, X., Xu, X., Xu, D., Zhang, T., Liu, Z., and Dong, Q. (2021). “Detection of concealed cracks from ground penetrating radar images based on deep learning algorithm.” Construction and Building Materials, 273, 121949.

Li, W., Cui, X., Guo, L., Chen, J., Chen, X., and Cao, X. (2016). “Tree root automatic recognition in ground penetrating radar profiles based on randomized hough transform.” Remote Sensing, 8(5).

Liu, H., Zheng, J., Yu, J., Xiong, C., Li, W., and Deng, J. (2023a). “Clustering of asphalt pavement maintenance sections based on 3d ground-penetrating radar and principal component techniques.” Buildings, 13(7).

Liu, Z., Gu, X., Chen, J., Wang, D., Chen, Y., and Wang, L. (2023b). “Automatic recognition of pavement cracks from combined gpr b-scan and c-scan images using multiscale feature fusion deep neural networks.” Automation in Construction, 146, 104698.

Liu, Z., Wang, S., Gu, X., Wang, D., Dong, Q., and Cui, B. (2024). “Intelligent assessment of

pavement structural conditions: A novel femvit classification network for gpr images.” IEEE Transactions on Intelligent Transportation Systems, 25(10), 13511–13523.

Liu, Z., Yeoh, J. K., Gu, X., Dong, Q., Chen, Y., Wu, W., Wang, L., and Wang, D. (2023c). “Automatic pixel-level detection of vertical cracks in asphalt pavement based on gpr investigation and improved mask r-cnn.” Automation in Construction, 146, 104689.

Lu, L., Pan, Y., Sheil, B., Luo, P., and Brilakis, I. (2026). “Regional graph-based pavement condition data quality assessment in support of trustworthy highway infrastructure digital twins.” Automation in Construction, 181, 106603.

Lu, L., Yin, M., Xie, Y., Pan, Y., Wang, M., and Brilakis, I. (2025). “Development of a trustworthy ai-supported digital twin framework for road operation and maintenance.” Smart Construction.

Luo, T. X., Lai, W. W., Chang, R. K., and Goodman, D. (2019). “Gpr imaging criteria.” Journal of Applied Geophysics, 165, 37–48.

Malihi, S., Potseluyko, L., Mathew, A., Alavi, H., Kumar Reja, V., Pan, Y., Binni, L., Wang, G., Wang, X., and Brilakis, I. (2024). “Review of multimodal data and their applications for road maintenance.” Smart Construction.

Namgyu Kim, Sehoon Kim, Y.-K. A. and Lee, J.-J. (2021). “A novel 3d gpr image arrangement for deep learning-based underground object classification.” International Journal of Pavement Engineering, 22(6), 740–751.

National Highways, UK. CS 229: Data for pavement assessment, <https://www.standardsforhighways.co.uk/tses/attachments/2e9e1b1c-528a-4b7d-bea8- d1c49a3caded?inline=true> (September 5, 2025).

National Highways, UK. CS 464: Non-destructive testing of highways structures, <https://www.standardsforhighways.co.uk/tses/attachments/f9ea394a-0d4b-4e64-8bf9- 73b330403832?inline=true> (September 5, 2025).

Pan, Y., Braun, A., Brilakis, I., and Borrmann, A. (2022a). “Enriching geometric digital twins of buildings with small objects by fusing laser scanning and ai-based image recognition.” Automation in Construction, 140, 104375.

Pan, Y., Noichl, F., Braun, A., Borrmann, A., and Brilakis, I. (2022b). “Automatic creation and enrichment of 3d models for pipe systems by co-registration of laser-scanned point clouds and photos.

Pan, Y., Wang, M., Lu, L., Wei, R., Cavazzi, S., Peck, M., and Brilakis, I. (2024). “Scan-to-graph: Automatic generation and representation of highway geometric digital twins from point cloud data.” Automation in Construction, 166, 105654.

Pengcheng Shangguan, Imad Al-Qadi, A. C. and Zhao, S. (2016). “Algorithm development for the application of ground-penetrating radar on asphalt pavement compaction monitoring.” International Journal ofPavement Engineering, 17(3), 189–200.

Pham, M.-T. and Lefèvre, S. (2018). “Buried object detection from b-scan ground penetrating radar data using faster-rcnn.” IGARSS 2018 - 2018 IEEE International Geoscience and Remote Sensing Symposium, 6804–6807.

Rasol, M., Pais, J. C., Pérez-Gracia, V., Solla, M., Fernandes, F. M., Fontul, S., Ayala-Cabrera, D., Schmidt, F., and Assadollahi, H. (2022). “Gpr monitoring for road transport infrastructure: A systematic review and machine learning insights.” Construction and Building Materials, 324, 126686.

Redmon, J. (2016). “You only look once: Unified, real-time object detection.” Proceedings of the IEEE conference on computer vision and pattern recognition.

Ren, Q., Wang, Y., and Xu, J. (2024). “A dl method to detect multi-type hidden objects in tunnel linings using a comprehensive gpr dataset.” Measurement, 238, 115379.

Solla, M., Pérez-Gracia, V., and Fontul, S. (2021). “A review of gpr application on transport infrastructures: Troubleshooting and best practices.” Remote Sensing, 13(4).

Su, Y., Wang, J., Li, D., Wang, X., Hu, L., Yao, Y., and Kang, Y. (2023). “End-to-end deep learning model for underground utilities localization using gpr.” Automation in Construction, 149, 104776.

Tong, Z., Gao, J., and Zhang, H. (2017). “Recognition, location, measurement, and 3d reconstruction of concealed cracks using convolutional neural networks.” Construction and Building

Materials, 146, 775–787.

Tong, Z., Gao, J., and Zhang, H. (2018). “Innovative method for recognizing subgrade defects based on a convolutional neural network.” Construction and Building Materials, 169, 69–82.

Tong, Z., Yuan, D., Gao, J., Wei, Y., and Dou, H. (2020). “Pavement-distress detection using ground-penetrating radar and network in networks.” Construction and Building Materials, 233, 117352.

Trimble Inc. Mobile mapping Trimble MX9, <https://geospatial.trimble.com/en/products/hardware/trimblemx9> (June 9, 2025).

Wang, J., Liu, H., Jiang, P., Wang, Z., Sui, Q., and Zhang, F. (2022). “Gpri2net: A deep-neuralnetwork-based ground penetrating radar data inversion and object identification framework for consecutive and long survey lines.” IEEE Transactions on Geoscience and Remote Sensing, 60, 1–20.

Wu, H. and Sheil, B. (2025). “Hybrid data generation and deep learning for gpr-based reconstruction of robotic-built underground structures.” Automation in Construction, 176, 106275.

Zhang, J., Yang, X., Li, W., Zhang, S., and Jia, Y. (2020). “Automatic detection of moisture damages in asphalt pavements from gpr data with deep cnn and irs method.” Automation in Construction, 113, 103119.

Zhao, S., Shangguan, P., and Al-Qadi, I. L. (2015). “Application of regularized deconvolution technique for predicting pavement thin layer thicknesses from ground penetrating radar data.” NDT & E International, 73, 1–7.

Zhou, S., Lin, J.-R., Pan, P., Pan, Y., and Brilakis, I. (2025). “Impact of color and mixing proportion of synthetic point clouds on semantic segmentation.” Automation in Construction, 171, 105963.

Zhou, Y., Zhang, J., Hu, Q., Zhao, P., Yu, F., Ai, M., and Huang, Y. (2024). “Deep learning based method for 3d reconstruction of underground pipes in 3d gpr c-scan data.” Tunnelling and Underground Space Technology, 150, 105819.

Zhu, Q. and Collins, L. (2005). “Application of feature extraction methods for landmine detection using the wichmann/niitek ground-penetrating radar.” IEEE Transactions on Geoscience and

Remote Sensing, 43(1), 81–85.

Zong, Z., Chen, C., Mi, X., Sun, W., Song, Y., Li, J., Dong, Z., Huang, R., and Yang, B. (2019). “A deep learning approach for urban underground objects detection from vehicle-borne ground penetrating radar data in real-time.” The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, XLII-2/W16, 293–299.

## List of Tables

1 Sample number of diferent classes in prepared dataset 38   
2 Prediction performance of various networks on test set (%) 39   
3 Classwise comparison for patch and crack classification (%) 40   
4 Ablation study of individual techniques on patch classification (%) . 41   
5 Results of evaluating diferent number of input layers on proposed model for patch   
classification (%) 42   
6 Results of evaluating diferent number of input layers on proposed model for crack   
classification (%) 43

TABLE 1. Sample number of diferent classes in prepared dataset
<table><tr><td>Class</td><td>Training Set</td><td>Test Set</td><td>Total</td></tr><tr><td>Patch</td><td>652</td><td>220</td><td>872</td></tr><tr><td>Crack</td><td>633</td><td>140</td><td>773</td></tr><tr><td>No defect</td><td>800</td><td>300</td><td>1100</td></tr><tr><td>Total</td><td>2085</td><td>660</td><td>2745</td></tr></table>

TABLE 2. Prediction performance of various networks on test set (%)
<table><tr><td>Defect</td><td>Model</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td rowspan="4">Patch</td><td>Resnet18</td><td>59.6</td><td>58.9</td><td>59.0</td><td>58.9</td></tr><tr><td>Resnet18 (pretrain)</td><td>59.3</td><td>59.4</td><td>59.4</td><td>58.8</td></tr><tr><td>3D Conv</td><td>67.6</td><td>66.9</td><td>65.8</td><td>65.9</td></tr><tr><td>Proposed network</td><td>69.0</td><td>68.2</td><td>68.0</td><td>68.1</td></tr><tr><td rowspan="4">Crack</td><td>Resnet18</td><td>54.9</td><td>52.4</td><td>52.7</td><td>51.8</td></tr><tr><td>Resnet18 (pretrain)</td><td>54.0</td><td>51.4</td><td>51.6</td><td>50.7</td></tr><tr><td>3D CNN</td><td>68.6</td><td>64.3</td><td>64.1</td><td>64.1</td></tr><tr><td>Proposed network</td><td>69.4</td><td>64.6</td><td>64.0</td><td>64.1</td></tr></table>

TABLE 3. Classwise comparison for patch and crack classification (%)
<table><tr><td>Defect type</td><td>Class</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td rowspan="3">Patch</td><td>Normal</td><td>72.5</td><td>75.7</td><td>74.1</td></tr><tr><td>Patch</td><td>64.7</td><td>60.9</td><td>62.8</td></tr><tr><td>Average</td><td>68.6</td><td>68.3</td><td>68.4</td></tr><tr><td rowspan="3">Crack</td><td>Normal</td><td>78.5</td><td>77.7</td><td>78.1</td></tr><tr><td>Crack</td><td>53.2</td><td>54.3</td><td>53.7</td></tr><tr><td>Average</td><td>65.8</td><td>66.0</td><td>65.9</td></tr></table>

TABLE 4. Ablation study of individual techniques on patch classification (%)
<table><tr><td>Model</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>3DCNN</td><td>67.6</td><td>66.9</td><td>65.8</td><td>65.9</td></tr><tr><td>Residual block</td><td>66.8</td><td>66.0</td><td>65.9</td><td>66.0</td></tr><tr><td>Depth attention</td><td>68.3</td><td>67.4</td><td>67.2</td><td>67.3</td></tr><tr><td>Channel attention</td><td>67.5</td><td>66.9</td><td>66.7</td><td>66.6</td></tr><tr><td>Mixed kernel size</td><td>66.7</td><td>65.9</td><td>65.8</td><td>65.8</td></tr><tr><td>Proposed network</td><td>69.0</td><td>68.2</td><td>68.0</td><td>68.1</td></tr></table>

TABLE 5. Results of evaluating diferent number of input layers on proposed model for patch classification (%)
<table><tr><td>Layer Number</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>5</td><td>67.5</td><td>66.8</td><td>65.9</td><td>65.9</td></tr><tr><td>10</td><td>68.1</td><td>67.3</td><td>67.0</td><td>67.1</td></tr><tr><td>15</td><td>68.5</td><td>67.7</td><td>67.5</td><td>67.5</td></tr><tr><td>20</td><td>69.0</td><td>68.2</td><td>68.0</td><td>68.1</td></tr></table>

TABLE 6. Results of evaluating diferent number of input layers on proposed model for crack classification (%)
<table><tr><td>Layer Number</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>5</td><td>66.1</td><td>60.6</td><td>55.5</td><td>53.4</td></tr><tr><td>10</td><td>70.6</td><td>65.8</td><td>64.1</td><td>64.4</td></tr><tr><td>15</td><td>70.0</td><td>65.5</td><td>65.5</td><td>65.4</td></tr><tr><td>20</td><td>69.0</td><td>68.2</td><td>68.0</td><td>68.1</td></tr></table>

## List of Figures

1 Equipment used in CAMHighways Dataset (d’Avigneau et al. 2025) 45   
2 Orthomosaics built from the pavement images and defect masks 46   
3 Illustration of spatial misalignment between orthomosaics and GPR data for the   
same road section 47   
4 Exemplar raw GPR data collected from two lanes of road section 48   
5 Extraction of overlapping area from example orthomosaics and GPR data 49   
6 Extracted orthomosaics and GPR data layers for one lane 50   
7 Overlapping area of orthomosaics and GPR coverage and corresponding fitted   
centerline 51   
8 Composite dataset of one lane divided into segments using one moving plane   
orthogonal to lane direction 52   
9 Visualisation of prepared data segments 53   
10 Proposed network architecture for GPR segment classification tasks 54   
11 RGB-GPR segments with annotations 55   
12 Training curves for proposed network 56   
13 Confusion matrix for patch and crack classification tasks 57

![](images/6b4caa743215dbb6b905b9b8d3c7d5036a9fa744575afafc70186346f500078f.jpg)

![](images/0a3b360b2d62138a2582a27bac514f76e00be991b421833cd955049609740bfc.jpg)  
(a) Trimble MX9 mobile mapping system (image (b) Hexagon Stream Up GPR system (Image source: source Trimble (Trimble Inc )) Hexagon (IDS GeoRadar ))  
Fig. 1. Equipment used in CAMHighways Dataset (d’Avigneau et al. 2025)

![](images/e166a69a34f5b62e81fc67a9e99140b2009e339e7711ff36f1d975ff0921c721.jpg)  
Fig. 2. Orthomosaics built from the pavement images and defect masks

![](images/e235e29ebb112bceb6b4a00f67c68d232d19dbac72644e5fb8d1b097bc0e17e3.jpg)  
(a) Exemplar orthomosaics of one road section

![](images/bcdabf51b24fa6c51c9d83441dc81c3832b4099f8f16d88bec2f1268864166f0.jpg)  
(b) Exemplar GPR data coverage for the same road section

![](images/6ae419945d3fe09e5073691c6fe6f20552ed43a6a3964e131355d74819949ba7.jpg)  
(c) Overlapped orthomosaics and GPR data  
Fig. 3. Illustration of spatial misalignment between orthomosaics and GPR data for the same road section

![](images/392f8df2c8cf7be10244de0f5f877a6cacfaf806aa275225da7dd32b01011224.jpg)

(a) 2D GPR data at the road surface (z = 0 cm)  
(b) 2D GPR data underground at a depth of z = -95 cm  
![](images/ccee11df1516572592399b3fb8c0b0df56054c8632c89ab8a1783ab60a6e3713.jpg)  
(c) 20 layers of GPR data structured in 3D  
Fig. 4. Exemplar raw GPR data collected from two lanes of road section

![](images/a5858e6f1cf42af998284a0a9363cac3ca9c67734e2f7210ced690ca5cf005af.jpg)  
Fig. 5. Extraction of overlapping area from example orthomosaics and GPR data

![](images/4b03da18f8cf745ed5859a33bd6c4e1a7e990050875b811c681cf7dce8388b80.jpg)  
Fig. 6. Extracted orthomosaics and GPR data layers for one lane

![](images/c2bf2561c0a2099d6e5af9af8a3985cd7f5a72e9eafa229c119067d9131d9415.jpg)  
Fig. 7. Overlapping area of orthomosaics and GPR coverage and corresponding fitted centerline

![](images/8233c70106f1abe157c6af80a7ef2d123ff561cb4ea86a57c14f4c739a8ba782.jpg)  
Fig. 8. Composite dataset of one lane divided into segments using one moving plane orthogonal to lane direction

![](images/0d873baee00906a8df9819be6cd569e5fa3e2bf213859f4db642904c1a1eecb6.jpg)  
Fig. 9. Visualisation of prepared data segments

![](images/4d2ade1e3f0e2813ef52f069c44a2a80628a459a276727aa77a667ce6e5ea513.jpg)  
Fig. 10. Proposed network architecture for GPR segment classification tasks

![](images/7d62cae64f07e07ff03b8ed945e174eaab09fdce7cf8d783276393d74bb3200d.jpg)  
(a) Patch segment

![](images/1cdcde9d7e9348a7706bacd94dbf72d415f5dcc5d75c2d889167bbc4302a60be.jpg)  
(b) Patch segment with annotation

![](images/b556142ca81507d56612cb277d43dc560f2a95e7cc62bb06f69917b03ea201a1.jpg)  
(c) Crack segment

![](images/da73aaf5e5f335a119881a60da1eb87492540dabbd25e853319f10606135054e.jpg)  
(d) Crack segment with annotation  
Fig. 11. RGB-GPR segments with annotations

![](images/a9778b7cf9fda8f0698f1025f844c97e8e706278eea27f8dc5c29773ad504ab1.jpg)

![](images/91601389c5adb778f2d42a1986a07eda90b74ea08b0f78dbdd044124ab457ecb.jpg)  
(a) Training accuracy curve for patch clas- (b) Training accuracy curve for crack classification sification  
Fig. 12. Training curves for proposed network

![](images/c154ce019f4eea16084cabe02bed393537708d2c680f415fb2945f4d072e5b4e.jpg)  
(a) Confusion matrix for patch classification

![](images/81448b51373834a700e6af7fc80054d83211913d5a2afed32fdf2b932087282f.jpg)  
(b) Confusion matrix for crack classification  
Fig. 13. Confusion matrix for patch and crack classification tasks