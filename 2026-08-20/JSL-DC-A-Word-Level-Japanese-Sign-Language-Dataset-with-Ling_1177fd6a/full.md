# JSL-DC: A Word-Level Japanese Sign Language Dataset with Linguist-Derived Descriptions for Distinguishing Confusable Signs

Ken Takaki<sup>1</sup> Asuka Ando<sup>1</sup> Misa Suzuki<sup>2</sup> Uiko Yano<sup>3</sup> Masaya Tsujimoto<sup>1</sup> Bill Neubauer<sup>4</sup> Ananay Vikram Gupta<sup>4</sup> Rose Shao<sup>5</sup> Matthias Hoppe<sup>5</sup> Sahir Shahryar<sup>4</sup> Celeste Mason<sup>4</sup> Kai Kunze<sup>6</sup> Yohei Oseki<sup>1</sup> Yoshihiro Kawahara<sup>1</sup> Thad Starner<sup>4</sup>

<sup>1</sup>The University of Tokyo <sup>2</sup>Independent <sup>3</sup>Kwansei Gakuin University <sup>4</sup>Georgia Institute of Technology <sup>5</sup>Keio University <sup>6</sup>Technical University of Clausthal

takaki@akg.t.u-tokyo.ac.jp

## Abstract

Effective sign language (SL) acquisition is crucial for deaf children, yet 95% are born to hearing parents who often lack proficiency in SL. SL recognition can power learning tools to help parents communicate with their children. However, Japanese Sign Language (JSL) lacks large-scale, multi-signer datasets, hindering the development of models that can generalize to new users. To address this gap, we introduce JSL-DC, the largest JSL dataset by video count, comprising 36.7K videos from 19 signers. The entire process was Deaf-centric: the lexicon comprising 270 JSL words was selected by Deaf and Coda linguists to facilitate parent-child communication, all participants were Deaf individuals who use JSL daily, and the data underwent a two-stage review process involving Deaf linguists. Moreover, we provide linguist-derived descriptions for distinguishing confusable signs. We demonstrate that the proposed model inspired by the descriptions outperforms stateof-the-art recognition methods by 9.8% on the confusable subset. The dataset, along with its linguistic description that inspires new models, will be released under a CC-BY 4.0 license to accelerate research in SL recognition.

## 1. Introduction

Sign language is a native language for many deaf or hard-ofhearing individuals, and its acquisition during early child hood is crucial for language development [30]. Parent-child communication is vital for this process. However, 95% of deaf children are born to hearing parents, who often struggle to communicate effectively as the parents do not know sign language [35]. Currently, learning sign language poses a significant barrier for these parents [38], partly because access to instructions is limited and often requires traveling long distances, sometimes more than an hour. Early language acquisition in children often begins with short utterances that combine pointing and single words [52]; therefore, allowing parents to even acquire a basic vocabulary, without full fluency, can meaningfully support early communication. While online videos offer self-study options, learning is considered more effective when combined with hands-on practice [50]. An isolated sign language recognition (ISLR) based “bubble bursting game” has been proposed as a method to facilitate hands-on practice of sign language words [50].

Large-scale ISLR datasets are available for many sign languages (Table 1), including American (PopSign ASL v1.0 [50]: 200K videos, WLASL [31]: 21K videos), British (BSL-1K [1]: 273K videos), Chinese (CSL500 [24]: 125K videos, NationalCSL-DP [26]: 134K videos), Australian (MM-WLAuslan [46]: 283K videos), and Turkish (AUTSL [48]: 38K videos) sign languages. Starting with datasets like WLASL [31], numerous ISLR models have been proposed. Consequently, recognition accuracy has reached practical levels for ASL [20, 21, 58], Auslan [46], CSL [24], and TSL [25], often exceeding 80% accuracy on classification tasks with over 100 glosses. A critical limitation remains, however. Some sign pairs share a handshape and differ only in non-manual cues, such as mouth or facial movements, or in how they are used in context. Datasets rarely label which pairs are confusable, and to our knowledge, none explain how to tell each pair apart, leaving computer vision researchers without linguistic guidance on these hard cases.

Moreover, datasets for Japanese Sign Language (JSL) are extremely scarce. One available dataset contains over 6,000 signs but is limited to a single signer [37], making it almost impossible to build ISLR models that generalize to new users. Another dataset includes 121 signers but only 100 signs, offering a very limited vocabulary [7]. As a result, the development of JSL recognition models has lagged significantly. Due to the reliance on single-signer data [37, 41], some studies fail to evaluate on unseen signers [3, 5]. Others that do perform unseen-signer evaluation use data from novice signers imitating sign motions [51], and their performance is expected to degrade when applied to native signers.

In this paper, we introduce JSL-DC, the largest JSL dataset by video count, exceeding the previous largest by 3 times and comprising 36.7K videos from 19 signers. It covers 270 signs with 136 repetitions per sign. We outline our contribution as follows.

• Deaf Centric data collection: the lexicon for facilitating the communication between hearing parents and deaf children was determined by Deaf and child of deaf adults (Coda<sup>1</sup>) linguists. Moreover, all participants in the data collection are Deaf individuals who use JSL daily, and the dataset’s quality is ensured by a two-stage review process conducted by reviewers raised by deaf parents and Deaf linguists.

• Distinguishing Confusable Signs by descriptions by Deaf linguists: To enhance research on distinguishing glosses that are confusable, JSL linguists annotate pairs of glosses that cannot be distinguished by handshape alone, and we add an explanation of how to distinguish each pair.

• Demonstrating less Confusion and 9.8% F1 score gain: Based on the descriptions added by the Deaf linguist, we resolve the confusion among glosses that are distinguished by mouth movement using AV-HuBERT used in lip-reading tasks. As a result, we show a further 9.8% F1 score gain on the confusable glosses over a state-of-theart model that already uses facial features, demonstrating that linguists’ descriptions of how to distinguish signs help guide improvements in sign language recognition.

We note that this dataset is not intended for training JSL-to-Japanese translation models, but rather for training recognition models for sign language learning tools; thus, we focused on collecting a specific variant of a sign. The dataset, including facial data, will be released under a Creative Commons CC-BY 4.0 license with full participant consent, which we believe will accelerate research in ISLR.

## 2. Related Work

Sign language recognition (SLR) is typically divided into two domains: continuous (CSLR) [21, 22, 33, 43, 54, 57] and isolated (ISLR) recognition [6, 20, 21, 23, 25, 33, 38, 56, 58], which focus on sentence-level and word-level videos, respectively.

ISLR is central for sign language learning applications because it provides the word-level recognition needed for vocabulary practice. To support JSL learning tools, this work introduces a new ISLR dataset for Japanese Sign Language. Accordingly, this section provides an overview of the ISLR landscape, covering datasets, recognition methodologies, and recent trends, with an emphasis on Japanese Sign Language (JSL).

## 2.1. Datasets for Isolated Sign Language Recognition

As shown in Table 1, various datasets for ISLR have been proposed worldwide. While large-scale datasets comprising over 100,000 videos exist for American (PopSign ASL v1.0 [50], PopSign ASL v2.0 [38]), British (BSL-1K [1]), Chinese (CSL500 [24], NationalCSL-DP [26]), and Australian (MM-WLAuslan [46]) sign languages, only NationalCSL-DP flags which of these are confusing signs that cannot be recognized from hand shape alone. To our knowledge, no dataset provides explicit linguist-authored distinguishing descriptions. Furthermore, in many datasets, the signers are novices who imitate signs, and annotation is rarely performed by Deaf linguists. It is therefore questionable whether existing datasets have correctly collected confusing signs in the first place.

Among existing JSL datasets, KoSign [37] offers a large vocabulary of 6,359 signs, but includes only a single signer, which limits the model’s ability to generalize to unseen signers. The Colloquial Corpus of Japanese Sign Language [7] features 121 signers but was primarily intended to map regional variations in JSL expressions, resulting in a limited vocabulary of 100. Furthermore, it is unsuitable for ISLR tasks because its videos often contain multiple repetitions of a sign or include unannotated, continuous conversations between Deaf signers.

In this work, we build the largest JSL dataset by video count, exceeding the previous largest by 3 times [7]. We asked 19 participants to record sign data using tablets with our collection app in their daily life environments. Our dataset is the first JSL dataset collected using mobile devices in such a natural setting. Furthermore, our dataset includes pairs of confusing signs that cannot be recognized from hand shape alone, together with linguists’ explanations of how to distinguish them. A distinguishing description is only useful if every clip in a confusable class shows the intended sign. Enforcing this is hard: even though all our participants use JSL daily, 9.8% of clips collected for the confusable glosses were reclassified as variants in review, their meaning differing from the reference or not produced correctly. Filtering these keeps each class clean, and their prevalence shows that review by Deaf signers is essential for confusable-sign data.

Table 1. Comparison of existing ISLR datasets. Distinguish Note is a description by a Deaf linguist for distinguishing confusable signs.
<table><tr><td>Dataset</td><td>Country</td><td>Signs</td><td>Signers</td><td>Videos</td><td>Ave. Videos/Sign</td><td>Source</td><td>Distinguish Note</td></tr><tr><td>Purdue RVL-SLLL [32]</td><td>USA</td><td>39</td><td>14</td><td>2.6K</td><td>67</td><td>Studio</td><td></td></tr><tr><td>RWTH-BOSTON 50 [55]</td><td>USA</td><td>50</td><td>3</td><td>0.5K</td><td>10</td><td>Studio</td><td></td></tr><tr><td>ASLLVD [4]</td><td>USA</td><td>3,000</td><td>6</td><td>9.8K</td><td>3</td><td>Studio</td><td></td></tr><tr><td>WLASL [31]</td><td>USA</td><td>2,000</td><td>119</td><td>21.1K</td><td>11</td><td>Web</td><td></td></tr><tr><td>MS-ASL [27]</td><td>USA</td><td>1,000</td><td>222</td><td>25.5K</td><td>26</td><td>Web</td><td></td></tr><tr><td>ASL Citizen [12]</td><td>USA</td><td>2,731</td><td>52</td><td>83.9K</td><td>31</td><td>Webcam</td><td></td></tr><tr><td>PopSign ASL v1.0 [50]</td><td>USA</td><td>250</td><td>47</td><td>200.7K</td><td>803</td><td>Phone</td><td></td></tr><tr><td>PopSign ASL v2.0 [38]</td><td>USA</td><td>562</td><td>76</td><td>363.7K</td><td>647</td><td>Phone</td><td></td></tr><tr><td>BOBSL [2]</td><td>GBR</td><td>2,281</td><td>39</td><td>452K</td><td>198</td><td>TV</td><td></td></tr><tr><td>BSL-1K [1]</td><td>GBR</td><td>1,064</td><td>40</td><td>273.0K</td><td>257</td><td>TV</td><td></td></tr><tr><td>BSLDict [34]</td><td>GBR</td><td>9283</td><td>148</td><td>14.2K</td><td>2</td><td>Studio</td><td></td></tr><tr><td>DEVISIGN-D [11]</td><td>CHN</td><td>500</td><td>8</td><td>6K</td><td>12</td><td>Studio</td><td></td></tr><tr><td>DEVISIGN-G [11]</td><td>CHN</td><td>36</td><td>8</td><td>432</td><td>12</td><td>Studio</td><td></td></tr><tr><td>DEVISIGN-L [11]</td><td>CHN</td><td>2,000</td><td>8</td><td>24.0K</td><td>12</td><td>Studio</td><td></td></tr><tr><td>CSL 500 [24]</td><td>CHN</td><td>500</td><td>50</td><td>125.0K</td><td>250</td><td>Studio</td><td></td></tr><tr><td>NationalCSL-DP [26]</td><td>CHN</td><td>6,707</td><td>10</td><td>134.1K</td><td>20</td><td>Studio</td><td></td></tr><tr><td>UWB-06-SLR-A [9]</td><td>CZE</td><td>25</td><td>15</td><td>5.5K</td><td>220</td><td>Studio</td><td></td></tr><tr><td>DGS Kinect 40 [40]</td><td>DEU</td><td>40</td><td>14</td><td>2.8K</td><td>70</td><td>Studio</td><td></td></tr><tr><td>SMILE [13]</td><td>DEU/CHE</td><td>100</td><td>30</td><td></td><td></td><td>Studio</td><td></td></tr><tr><td>GSL 982 [40]</td><td>GRC</td><td>982</td><td>1</td><td>4.9K</td><td>5</td><td>Studio</td><td></td></tr><tr><td>INCLUDE [49]</td><td>ISR</td><td>263</td><td>7</td><td>4.3K</td><td>16</td><td>Studio</td><td></td></tr><tr><td>KL_MV2DSL [36]</td><td>ISR</td><td>200</td><td>5</td><td>5.0K</td><td>25</td><td>Studio</td><td></td></tr><tr><td>LSA64 [44]</td><td>ARG</td><td>64</td><td>10</td><td>3.2K</td><td>50</td><td>Studio</td><td></td></tr><tr><td>LSE-Sign [17]</td><td>ESP</td><td>2,400</td><td>2</td><td>2.4K</td><td>1</td><td>Studio</td><td></td></tr><tr><td>LSFB-ISOL [15]</td><td>FRA/BEL</td><td>395</td><td>100</td><td>47.6K</td><td>121</td><td>Studio</td><td></td></tr><tr><td>Logos [42]</td><td>RUS</td><td>2,863</td><td>381</td><td>200K</td><td>70</td><td>Phone &amp; Webcam</td><td></td></tr><tr><td>Slovo [29]</td><td>RUS</td><td>1,000</td><td>194</td><td>20K</td><td>20</td><td>Phone &amp; Webcam</td><td></td></tr><tr><td>Bosphorus Sign [8]</td><td>TUR</td><td>855</td><td>10</td><td>51.3K</td><td>60</td><td>Studio</td><td></td></tr><tr><td>Bosphorus Sign22K [59]</td><td>TUR</td><td>744</td><td>6</td><td>22.5K</td><td>30</td><td>Studio</td><td></td></tr><tr><td>AUTSL [48]</td><td>TUR</td><td>226</td><td>43</td><td>38.3K</td><td>169</td><td>Studio</td><td></td></tr><tr><td>Auslan-Daily [45]</td><td>AUS</td><td>13,945</td><td>67</td><td>25.1K</td><td>2</td><td>Web</td><td></td></tr><tr><td>MM-WLAuslan [46]</td><td>AUS</td><td>3,215</td><td>73</td><td>282.9K</td><td>88</td><td>Studio</td><td></td></tr><tr><td>Colloquial Corpus of JSL [7]</td><td>JPN</td><td>100</td><td>121</td><td>12.1K</td><td>121</td><td>Studio</td><td></td></tr><tr><td>KoSign [37]</td><td>JPN</td><td>6359</td><td>1</td><td>6.4K</td><td>1</td><td>Studio</td><td></td></tr><tr><td>A Guide to Hand Gestures and Words [41]</td><td>JPN</td><td>469</td><td>1</td><td>469</td><td>1</td><td>Studio</td><td></td></tr><tr><td>Ours (reference videos)</td><td>JPN</td><td>270</td><td>2</td><td>540</td><td>2</td><td>Studio</td><td>√</td></tr><tr><td>Ours</td><td>JPN</td><td>270</td><td>19</td><td>36.7K</td><td>136</td><td>Tablet</td><td>√</td></tr></table>

## 2.2. Isolated Sign Language Recognition Methods

ISLR involves classifying the gloss label of a sign. This field is broadly divided into three main approaches: posebased, pixel-based, and hybrid methods.

Pose-based ISLR methods first extract skeletal information using tools such as MediaPipe [16] and then process these features with a recognition model [6, 23, 33, 38, 56]. Since these methods extract low-dimensional skeletal features using lightweight models, they are generally fast at inference and robust to variations in skin tone and illumination [20]. However, pose estimators like MediaPipe struggle with challenges, such as low accuracy in depth estimation and difficulties with hand occlusions, both of which can lead to substantial performance degradation [19].

In contrast, pixel-based ISLR processes raw video frames directly using recognition models such as CNNs [10, 53]. While these methods are computationally intensive, they tend to achieve higher accuracy in distinguishing between visually similar signs [20]. Recently, hybrid approaches that combine pose-based and pixel-based features have been introduced to further improve performance [20, 21, 23, 25, 58].

ISLR for JSL has often been constrained by datasets limited to a single signer [37, 41]. Consequently, some studies do not evaluate model performance on unseen signers [3, 5]. Although other studies have evaluated models on unseen signers, they relied on data of novice signers imitating JSL signs [51]. This approach presents a significant limitation, as the spatial and temporal characteristics of nonnative signing differ significantly from those of native signers.

In contrast, our dataset directly addresses these limitations. It features 19 participants, all of whom are Deaf individuals who use JSL in their daily lives. Furthermore, the reviews were performed by Deaf experts, including those with expertise in sign linguistics. We believe this dataset provides a strong foundation for training ISLR models that can generalize to unseen native signers, a critical and previously unmet requirement for real-world applications.

## 3. Collection Methodology

We adopted a Deaf-centric approach to ensure that the dataset meaningfully serves the needs and practices of the Deaf community. This “by the community, for the community” approach is crucial for developing technology that accurately reflects its target users. Our data collection process was led by two Deaf authors (D1 and D2) and one Coda author (C), who are experts in Deaf education or sign language linguistics. D1 and D2 grew up in families with a high proportion of Deaf members. D1 has professional experience working with deaf children in a preschool setting and maintains daily interaction with deaf people. D2 possesses extensive experience teaching JSL to hearing students. C has professional experience teaching English to deaf children at a school for the Deaf. Moreover, participants were required to be deaf adults aged 18 or older who use JSL in their daily lives.

The data collection process is illustrated in Figure 1. First, we selected the lexicon and then shot reference videos to instruct participants on which signs to perform. Next, D1 or D2 explained the data collection procedure to participants in JSL. The explanation included the data collection process using tablet devices and the associated risks of releasing the data, including the faces of individuals, to the public. Participants took the tablet devices home to record data, which was automatically uploaded to the cloud, and returned the tablets upon completion of the data collection. To ensure the quality and integrity of our dataset, all collected sign language videos underwent a two-stage review process conducted by Deaf or Coda individuals. We detail each step in the following sections.

## 3.1. Lexicon Selection and Shooting Reference Videos

The lexicon selection team (D1, D2, and C) selected approximately 270 JSL words, which is 2.7 times larger than the previous largest dataset with multiple signers [7].

Purpose of our lexicon. Different from other ISLR datasets, our lexicon aimed to create a priority list for hearing parents striving to establish effective communication with their deaf children. Therefore, we had to create our own lexicon instead of adapting the lexicons of other sign languages. It has been reported [52] that after one year of age, children begin to produce two-word sentences by combining pointing with a sign (e.g., pointing + verb/adjective or verb/adjective + pointing), which corresponds to a pronoun + verb/adjective structure. Here, pointing is used as a pronoun in sign language acquisition. Therefore, the acquisition of verbs and adjectives, which cannot be easily substituted by pointing, should be prioritized over nouns. As a result, our lexicon prioritizes verbs, adjectives, and greeting expressions that the researchers identified as highfrequency items for deaf children.

Extracting Japanese words from three sources. We referenced three sources for the lexicon. From the 103 words listed under Action words in the MacArthur-Bates Communicative Development Inventories [14], the lexicon selection team translated them into Japanese. For words such as “open,” which have distinct intransitive (開く, aku) and transitive (開 け る, akeru) forms in Japanese, both variants were included in the translation. Daily routines and greetings, action words, temporal expressions, and descriptive/attributive words were chosen from the Japanese MacArthur-Bates Communicative Development Inventories (CDI) [39]. In addition, verbs were extracted from firstgrade Japanese language textbooks [28]. The list resulted in 200 Japanese words, which were selected based on their frequent usage among Japanese deaf children, as anticipated by the lexicon selection team.

Japanese to JSL translation. Finally, for approximately 200 Japanese vocabulary items, D1 and D2 primarily led the process of identifying corresponding 270 JSL expressions. For verbs such as “wash” (洗う, arau), where the sign varies depending on the direct object, 2-3 sign variants were selected. Consequently, multiple JSL expressions may correspond to a single Japanese word. Additionally, JSLspecific expressions such as $( \ X - , \mathring { \mathfrak { i } } - , o \ – b a - )$ were incorporated into the list. $( \sharp - \ d ) \mathfrak { i } - , o { \ - b } a - )$ carries contextdependent meanings but is frequently used in interactions with young children to express surprise, equivalent to expressions such as “Really?” or “Wow, amazing!” Additionally, the Japanese word for writing has signs corresponding to vertical and horizontal writing.

## 3.2. Shooting Reference Videos

The process. We recorded D1 and D2 signing reference videos for the entire lexicon in a studio. Another Deaf individual trimmed the video’s duration. Finally, the authors discussed and selected the clearer version for all 270 vocabulary items as reference videos (Figure 2).

Apparatus. We shot videos against a green screen using a video camera (Panasonic Lumix DC-GH7L) at 4K 60 fps. Lighting was positioned to minimize facial shadows during signing, and signers wore plain black T-shirts to ensure that their hand shapes were clearly visible.

## 3.3. In-person Participation Introduction in JSL

We held an in-person introduction in JSL, led by D1 or D2, to ensure that participants understood the data collection process using tablets and the associated risks of releasing the data, including the faces of individuals, to the public.

After obtaining consent in JSL, each participant was asked to perform 270 words 20 times each, totaling 5400 examples, at their favorite place and time. They were informed that they would receive 75,000 yen in compensation for performing 5,400 signs, with compensation provided proportionally based on the number of completed signs if they quit earlier. To prevent leakage of residential location and other personal information during data publication, we instructed participants to ensure that nothing revealing such information appeared in the background and that only the participant themselves was visible in the recordings. Additionally, we requested that participants keep their hands within the camera’s field of view throughout the recording. Participants contacted D1, D2, or C if they had questions or problems during the collection period.

![](images/2af71a047e4d95d5ff52fbdc87e8c44f0fc0555085f8ddc79e1869d263c940c6.jpg)  
Figure 1. Overview of our Deaf-centric process of data collection involving deaf participants. Deaf authors D1, D2, and the Coda author C led the process.

![](images/04686cb2ee253ab86f1f23cad1c56d5954750145aeb9db949bb6c02908a82922.jpg)  
Figure 2. Sample reference videos used in the data collection. (a) is “scary” (怖い, kowai), and (b) is “cold” (寒い, samui), with the same body movement but different mouth movements.)

## 3.4. JSL Recorder App

Recording was performed using an open-source sign language recording application installed into a Google Pixel Tablet with a built-in RGB camera.

The video resolution is 2560 x 1920 pixels, and the frame rate is 30 fps with an average bitrate of 15 Mbps.

From the app’s home screen, pressing the session start button transitions to the sign recording screen, which displays the corresponding Japanese word, the participant’s video in the center, and the reference video on the right, as shown in Figure 3. D1 and D2 verified that the reference video is sufficiently large (8.3 × 7.8 cm on the tablet) for deaf participants to understand which sign to perform.

![](images/22eb6f6c3a4ff9bfc35f7101bcbac7ffb0717f9e866ebbe7f11ef4efefa8c816.jpg)  
Figure 3. Screenshot of the JSL recorder tablet application. The participant swipes to the left after finish signing each example.)

Participants could check the reference video, press the start button on screen, perform the sign, and swipe left when finished to proceed to the next sign. If participants wanted to redo a sign, they could press the restart button to retry. To maintain high signing quality, we limited each session to 30 examples and ensured that the same sign did not appear more than once within a single session.

Since tablet devices cannot instantly start recording, the recording began at the start of the session. We separately recorded timestamps for when the start button was pressed and when participants swiped after completing each sign, marking the start and end times of each sign. Based on these records, we segmented each sign in the video.

We found that the 5400th example (the sign corresponding to “call someone”) could not be recorded due to an application bug. This issue was experienced by all participants who finished all recordings.

## 3.5. Review Procedure

To ensure the quality and integrity of our dataset, all collected sign language videos underwent a rigorous, twostage review process conducted by native signers and linguistic experts. This pipeline was designed to filter out erroneous data, protect privacy, and categorize valid sign variations.

Two-Stage review pipeline. The review process was divided into two check steps. The entire review team performed an initial quality assessment to filter the collected signs. Signs that passed the initial check (or were flagged as “Unsure” ) were escalated to a 2nd check. This stage was conducted by Deaf linguists inside the review team.

As shown in Figure 3, our data collection app presents participants with a corresponding Japanese gloss and a reference video simultaneously. In this way, participants might perform a different sign from the one shown in the reference video, just by looking at the Japanese gloss, which is a common challenge in collecting sign language datasets [18]. Our review process with Deaf linguists allowed us to filter the dataset, retaining only the signing words that have the same meaning as those shown in the reference video.

## 4. Dataset Composition

## 4.1. Statistics

Twenty Deaf individuals who use JSL daily participated in data collection by snowball sampling. Still, one was excluded from the dataset due to frequent unnatural signing identified during the review process. Therefore, the dataset contains data from a total of 19 individuals, consisting of 11 males and 8 females, ranging in age from 18 to the 70s. The dataset was collected in Japan with an entirely Japanese participant pool.

Participants recorded videos in various locations and at different times of day, resulting in diverse backgrounds and lighting conditions. Our dataset comprises 36,705 videos that passed our two-stage review pipeline. Surprisingly, 9.8% of clips in the confusable glosses were removed as variants: the meaning differs from the reference signing, suggesting that a careful review is necessary to keep the samples for each gloss clean. The sign corresponding to the word “get angry” had the most data (309 samples), while the sign corresponding to the word “think” had the fewest (27 samples). We will also publicly release the reference videos used during data collection, an example of which is shown in Figure 2.

We split the dataset into train, validation, and test sets. The signers were divided into 15 (train), 2 (val), and 2 (test), with gender balance maintained across splits. This technique ensures a disjoint signer setup (i.e., individuals in one split do not appear in any other). All 270 sign classes are represented in each split. The resulting data proportions are 81.3% (train), 8.4% (validation), and 10.3% (test).

## 4.2. Descriptions for Distinguishing Confusable Signs

We provide additional descriptions to distinguish confusable signs and guide researchers in creating classifiers for them. D2 added descriptions for pairs that cannot be distinguished by handshape alone, together with an explanation of how to distinguish each pair. C verified the descriptions.

![](images/39cfa8f4877460e08a31551294172abde19f9cca1cc2783d2dcec7d97737696a.jpg)

Figure 4. Frequently confused examples. (a) is “bitter (taste)” (苦 い, nigai), and was misclassified as “spicy” (辛い, karai), which is (b). They could be classified by the mouth movements.  
![](images/2ac8300d36115263cb4e5006ee984fdafc68ccbe2602f73eaa3aa0d2ecaf662f.jpg)  
Figure 5. The proposed mouthing-based reranker to classify the confusable sets.

Twenty-two glosses belong to this set, differing in aspects such as mouthing or eye movement. Two groups rely partly on context, however, most pairs in our glosses can be distinguished by a Deaf person without the surrounding context.

We illustrate challenging and easily confused examples from our dataset in Figure 4. For instance, (a) shows the sign “bitter” (taste), and (b) is an example of “spicy”. They could be distinguished by their mouth movements.

Using these descriptions, sign language recognition researchers can refine the encoder they use or guide the model’s attention. We consider such labeling a useful guide for improving accuracy even with limited data, and we present an example of its use in Sec. 5, demonstrating that it improves recognition accuracy.

## 5. Experiments

To clarify the characteristics and challenges of our dataset, we evaluated the recognition accuracy using existing ISLR models. We selected I3D [10] and NLA-SLR (RGB input) [58] as a pixel-based method and SPOTER [6], Pop-Sign ASL v2.0 [38], MASA [56], and NLA-SLR (keypoint input) [58] as pose-based methods. Table 2 reports the Top-1 accuracy and F1 score over the 270 glosses, together with the scores on the 22 confusable glosses determined by the linguists and the remaining 248 glosses. Although almost every pair in our dataset could be distinguished by a Deaf person without the surrounding context, we found that even state-of-the-art ISLR models, which achieve an average accuracy above 90%, still struggle to distinguish the glosses in the confusable set.

As an example that leverages the Deaf linguist’s description of pairs that cannot be distinguished by handshape alone, we also propose the model shown in Fig. 5, which adds a mouthing classifier to NLA-SLR. To extract mouthmovement features, we used AV-HuBERT (Audio-Visual

<table><tr><td rowspan="2"></td><td rowspan="2">input</td><td rowspan="2">MACs (G)</td><td colspan="2">All (270 glosses)</td><td colspan="2">Confusable (22 glosses)</td><td colspan="2">Others (248 glosses)</td></tr><tr><td>Top-1</td><td>F1</td><td>Top-1</td><td>F1</td><td>Top-1</td><td>F1</td></tr><tr><td>SPOTER [6]</td><td>keypoint</td><td>0.38</td><td>78.7</td><td>78.5</td><td>51.7</td><td>50.9</td><td>81.1</td><td>80.9</td></tr><tr><td>PopSign ASL v2.0 [38]</td><td>keypoint</td><td>1.6</td><td>81.9</td><td>81.4</td><td>55.3</td><td>50.1</td><td>84.3</td><td>84.2</td></tr><tr><td>MASA [56]</td><td>keypoint</td><td>3.9</td><td>82.8</td><td>83.0</td><td>54.8</td><td>51.1</td><td>85.3</td><td>85.8</td></tr><tr><td>NLA-SLR [58]</td><td>keypoint</td><td>55.7</td><td>91.6</td><td>91.9</td><td>61.2</td><td>58.1</td><td>94.3</td><td>94.9</td></tr><tr><td>NLA-SLR + mouthing (ours)</td><td>keypoint</td><td>58.0</td><td>92.2</td><td>92.5</td><td>68.3</td><td>65.4</td><td>94.3</td><td>94.9</td></tr><tr><td>I3D [10]</td><td>RGB</td><td>111.2</td><td>83.5</td><td>83.1</td><td>54.1</td><td>47.7</td><td>86.1</td><td>86.2</td></tr><tr><td>NLA-SLR [58]</td><td>RGB</td><td>71.9</td><td>91.9</td><td>92.1</td><td>57.6</td><td>53.5</td><td>94.9</td><td>95.5</td></tr><tr><td>NLA-SLR + mouthing (ours)</td><td>RGB</td><td>74.7</td><td>92.5</td><td>92.9</td><td>65.7</td><td>63.3</td><td>94.9</td><td>95.5</td></tr></table>

Table 2. Results across confusable vs. other glosses.

Hidden Unit BERT) [47], which can perform the lip-reading task. Among the 22 confusable glosses, 16 glosses were identified as distinguishable by mouth movement according to the linguist-derived description. During training, we built one SVM per confusable pair among the 16 glosses to discriminate within each confusable combination based on mouth movement. At test time, when the predicted sign from the ISLR model is one of the 16 glosses, we extract mouth-region features with AV-HuBERT, classify with the SVM for that pair, and produce the final output.

We found that this raises the overall accuracy over NLA-SLR, which is typically the most accurate among existing models, while improving the F1 and Top-1 scores, especially for confusable glosses. Tab. 3 shows the change in Top-1 accuracy and F1 score when mouthing is added to NLA-SLR (RGB). These results show that some confusable signs improve in score, while others do not improve. For the other 254 glosses, there was no change in accuracy or F1 score. A Wilcoxon signed-rank test over the 270 glosses showed that our model achieved a significantly higher F1 score than NLA-SLR (p = 0.009 for the keypoint-based model and p = 0.008 for the RGB-based model).

Therefore, we found that mouth-movement features from AV-HuBERT alone can increase accuracy on the confusable glosses. These results suggest that linguist-derived descriptions can be used to build better models, and incorporating mechanisms that attends to other cues such as eye movements and hand speed may further improve performance.

## 6. PopSign JSL

We are currently working with the Deaf community to determine which types of games might be most compelling for learning JSL vocabulary. However, a local Deaf school is interested in a port of the open source PopSign ASL game [50] to help their Japanese hearing parents of Deaf children learn how to better communicate with their children in JSL. In addition, the Deaf school has requested a version of Pop-Sign to help their ninth-grade Deaf class learn written English vocabulary. The intention is that the children will sign the JSL translation of a concept when cued with a written English word.

<table><tr><td>Gloss</td><td>∆Top-1</td><td>∆F1</td></tr><tr><td>Thank you for the meal. (s0006_itatakimasu)</td><td>-19.0</td><td>-3.5</td></tr><tr><td>Thank you for the meal. (s0033_kochisousamateshita)</td><td>+31.2</td><td>+38.5</td></tr><tr><td>scared (s0034_kowai_Bu_i_)</td><td>+37.5</td><td>+33.8</td></tr><tr><td>cold (s0036_samui_Han_i_)</td><td>+10.0</td><td>+15.5</td></tr><tr><td>cold (s0057_tsumetai_Leng_tai_)</td><td>+44.4</td><td>+43.9</td></tr><tr><td>spicy (s0024_karai_Xin_i_)</td><td>0.0</td><td>+0.8</td></tr><tr><td>bitter (s0064_nikai.Ku.i-)</td><td>+11.1</td><td>+20.0</td></tr><tr><td>laugh (s0091_warau_Xiao_u_)</td><td>0.0</td><td>0.0</td></tr><tr><td>arrive (s0056.tsuku_Zhao_ku.)</td><td>+12.5</td><td>+24.6</td></tr><tr><td>stop (s0058_tomaru_Zhi_maru_)</td><td>+50.0</td><td>+41.8</td></tr><tr><td>pour (s0047_sosoku)</td><td>0.0</td><td>0.0</td></tr><tr><td>pour (s0050_sosoku_Zhu_ku_)</td><td>0.0</td><td>0.0</td></tr><tr><td>gather (s0039_shiyuukousuru_Ji_He_suru_)</td><td>0.0</td><td>0.0</td></tr><tr><td>small (s0052_chiisai_Xiao_sai_)</td><td>0.0</td><td>0.0</td></tr><tr><td>happy (s0010_ureshii_Xi_shii_)</td><td>0.0</td><td>0.0</td></tr><tr><td>fun (s0052_tanoshii_Le_shii_)</td><td>0.0</td><td>0.0</td></tr></table>

Table 3. Change in Top-1 accuracy and F1 (percentage points) after the mouthing rerank, for confusable pairs distinguished mainly by mouthing/mouth cues.

PopSign JSL is similar to Taito’s popular 1994 Japanese arcade game Puzzle Bobble, known internationally as Bust-A-Move. In the original game, players are presented with a field of colored balls. Players are provided with a colored ball at each turn, which they must release into the field of balls. Whenever three or more balls of the same color touch, they burst and disappear from the screen. The player attempts to remove all balls from the screen. PopSign JSL modifies this game mechanic in that the player can select the color of the ball they wish to release by signing the JSL sign associated with that color. Each color ball is labeled with a written concept that the player must match with their signing. Poor signing results in the wrong color ball being released.

Figure 6 shows a playable version of PopSign JSL designed for a smartphone. While many recognition systems tested here are not suitable for smartphones, the 3-layer Bi-LSTM recognizer runs in real time and achieves sufficient accuracy for gameplay. For hearing parents learning JSL, the bubbles are labeled with Japanese text. For the Japanese Deaf students learning English text, the balls are labeled in English. Future work includes user testing and iteration of the game, as well as increasing the vocabulary size.

![](images/05447a01224e72cf2b9fd1618e1d02006a4e14d76d2e1abd776ed7de623ff464.jpg)  
Figure 6. A screenshot of the playable version of PopSign JSL, where (a) the user is signing “warm” (暖かい, atatakai) and hand landmarks are recognized by MediaPipe (only the left hand is visualized). The landmarks are fed into a Bi-LSTM model trained with our dataset. (b) The bubble labeled with the recognition result is released.

## 7. Ethics and Safety Discussion

This research was conducted with Deaf researchers at the core of the team, maintaining constant awareness of the Deaf community. This research, including the dataset publication, was conducted after receiving approval from the institutional review board of the authors’ organizations. Participants received 75,000 yen in compensation for performing 5,400 signs, which took an average of 8.7 hours. Four participants did not complete the data collection, but compensation was provided based on the number of completed signs, and their videos were included in the dataset.

The dataset will be released under a Creative Commons CC-BY 4.0 license, allowing both academics and industry to use it. A key agreement of this license is that any use must not infringe upon the participants’ moral rights, particularly any application that is prejudicial to their honor or reputation. Participants retain the right to withdraw their data at any time. They can have their videos removed from our servers at any time by signing and submitting the withdrawal of consent form provided to them. However, we have explicitly informed participants that we cannot enforce the deletion of data that may have been downloaded by third parties after the dataset’s public release at camera-ready. Our two-stage review process rejected videos that contained privacy issues, such as visible personal information.

While the signs in the reference videos for this collected data were selected through discussion to represent the most commonly used forms among various sign variants, it should be noted that this dataset should not be used for establishing a standard for JSL, though there is a risk users might use it in that way.

## 8. Limitations and Future Work

Coverage of JSL variations. JSL varies by region and community, such as among different schools and communities. However, this dataset does not aim to cover all such variations, as our first application is not JSL-to-Japanese translation. Rather, our first future application is a sign language learning app, designed to teach one variant of a sign to facilitate communication between hearing parents and deaf children. Another application of ISLR would be helping JSL signers look up Japanese words in JSL to help them learn Japanese, which is a second language for them. In these cases, it would be desirable to support various JSL styles, including dialectal signs.

Word-level lexicon. This research focuses on individual words, especially verbs and adjectives, as they are essential for communication between hearing parents and deaf children in their early stages, as discussed in Section 3.1. Future collection with more nouns, phrases, or sentences will be necessary to create JSL-Japanese translation models.

Data biases. All participants in this dataset collection were Japanese. As a result, the dataset primarily represents a single ethnic group and does not capture a wide range of skin tones. Consequently, pixel-based ISLR methods trained on this data may struggle to generalize to different skin tones. However, pose-based methods are expected to be robust to such variations. Furthermore, participants were recruited via snowball sampling, which means we cannot rule out potential signing biases (e.g., community-specific variations) introduced by this sampling method. Finally, individuals under 18 are not included in this dataset due to the difficulties associated with obtaining consent for the public release of data that includes their faces.

## 9. Conclusion

In this paper, we introduced JSL-DC, the largest JSL dataset by video count for ISLR to date, with linguist-derived descriptions to distinguish confusable signs. Our experiments showed that our model, inspired by the description, improved the F1 score from 53.5% to 63.3% compared to the state-of-the-art method on our confusable glosses. We believe this data will enable various applications, starting with sign language learning for families with deaf children to support their language development, and ultimately bring a positive impact to the Deaf community in Japan.

## References

[1] Samuel Albanie, Gul Varol, Liliane Momeni, Triantafyllos¨ Afouras, Joon Son Chung, Neil Fox, and Andrew Zisserman. BSL-1K: Scaling up co-articulated sign language recognition using mouthing cues. In ECCV, pages 35–53, 2020. 1, 2, 3

[2] Samuel Albanie, Gul Varol, Liliane Momeni, Hannah Bull,¨ Triantafyllos Afouras, Himel Chowdhury, Neil Fox, Bencie Woll, Rob Cooper, Andrew McParland, and Andrew Zisserman. BBC-Oxford British Sign Language dataset. arXiv:2111.03635 [cs.CV], 2021. 3

[3] Yoshitaka Ando and Rei Kawakami. Japanese Sign Language Word Recognition using Video Processing Neural Networks. In the 86th National Convention of the IPSJ, pages 527–528, 2024. 2, 3

[4] Vassilis Athitsos, Carol Neidle, Stan Sclaroff, Joan Nash, Alexandra Stefan, Quan Yuan, and Ashwin Thangali. The American sign language lexicon video dataset. In CVPRW, pages 1–8, 2008. 3

[5] Bingyuan Bai and Kazunori Miyata. Isolated Japanese sign language recognition based on image entropy variation rate and score-level multi-cue fusion. In CGIP, pages 30–37, 2024. 2, 3

[6] Matya´s Bohˇ a´cek and Marek Hrˇ uz. Sign Pose-Based Trans-´ former for Word-Level Sign Language Recognition. In WACV, pages 182–191, 2022. 2, 3, 6, 7

[7] M Bono, Kouhei Kikuchi, Paul Cibulka, and Y Osugi. A colloquial corpus of Japanese sign language: Linguistic resources for observing sign language conversations. In LREC, pages 1898–1904, 2014. 2, 3, 4

[8] Necati Cihan Camgoz, A Kındıroglu, Serpil Karab ¨ ukl ¨ u,¨ Meltem Kelepir, A S Ozsoy, and L Akarun. BosphorusSign:<sup>¨</sup> A Turkish sign language recognition corpus in health and finance domains. In LREC, pages 1383–1388, 2016. 3

[9] Pavel Campr, Marek Hruz, and Milo ´ sˇ Zelezn <sup>ˇ</sup> y. Design and ´ recording of Czech sign language corpus for automatic sign language recognition. In Interspeech, pages 678–681. ISCA, 2007. 3

[10] Joao Carreira and Andrew Zisserman. Quo Vadis, action recognition? A new model and the kinetics dataset. In CVPR, pages 4724–4733, 2017. 3, 6, 7

[11] Xiujuan Chai, G Li, Yushun Lin, Zhihao Xu, Y Tang, and Xilin Chen. Sign language recognition and translation with Kinect. In ICAISC, pages 394–402, 2012. 3

[12] Aashaka Desai, Lauren Berger, Fyodor O Minakov, Vanessa Milan, Chinmay Singh, Kriston Pumphrey, Richard E Ladner, Hal Daum’e, Alex X Lu, Naomi K Caselli, and Danielle Bragg. ASL Citizen: A community-sourced dataset for advancing Isolated Sign Language Recognition. In NeurIPS, 2023. 3

[13] Sarah Ebling, Necati Cihan Camgoz, P B Braem, Katja Tissi,¨ Sandra Sidler-Miserez, Stephanie Stoll, Simon Hadfield, T Haug, R Bowden, Sandrine Tornay, Marzieh Razavi, and M Magimai-Doss. SMILE Swiss German sign language dataset. In LREC, pages 4221–4229, 2018. 3

[14] Larry Fenson, Virginia A Marchman, Donna J Thal, Philip S Dale, J Steven Reznick, and Elizabeth Bates. MacArthur-

Bates communicative development inventories. Paul H. Brookes Publishing Company, Baltimore, 2007. 4

[15] Jerome Fink, Benoit Frenay, Laurence Meurant, and Anthony Cleve. LSFB-CONT and LSFB-ISOL: Two new datasets for vision-based sign language recognition. In IJCNN, pages 1–8, 2021. 3

[16] Google. Google AI Edge. https://ai.google.dev/ edge / mediapipe / solutions / vision / pose \_ landmarker?hl=ja, 2025. Accessed: 2025-11-6. 3

[17] Eva Gutierrez-Sigut, Brendan Costello, Cristina Baus, and Manuel Carreiras. LSE-Sign: A lexical database for Spanish Sign Language. Behavior research methods, 48(1):123–137, 2016. 3

[18] J Hochgesang, O Crasborn, and Diane C Lillo-Martin. Building the ASL signbank. Lemmatization principles for ASL. In LREC, pages 69–74, 2018. 6

[19] Ruth M Holmes, Ellen Rushe, and Anthony Ventresque. The Key Points: Using Feature Importance to Identify Short comings in Sign Language Recognition Models. In LREC-COLING, pages 15970–15975, 2024. 3

[20] Marek Hruz, Ivan Gruber, Jakub Kanis, Maty´ a´s Bohˇ a´cek,ˇ Miroslav Hlava´c, and Zdenˇ ek Krˇ noul. One model is notˇ enough: Ensembles for isolated sign language recognition. Sensors, 22(13):5043, 2022. 1, 2, 3

[21] Hezhen Hu, Weichao Zhao, Wengang Zhou, and Houqiang Li. SignBERT+: Hand-model-aware self-supervised pretraining for sign language understanding. IEEE transactions on pattern analysis and machine intelligence, 45(9):11221– 11239, 2023. 1, 2, 3

[22] Lianyu Hu, Liqing Gao, Zekang Liu, and Wei Feng. Continuous sign language recognition with correlation network. In CVPR, pages 2529–2539, 2023. 2

[23] Lianyu Hu, Liqing Gao, Zekang Liu, and Wei Feng. Dynamic Spatial-Temporal Aggregation for Skeleton-Aware Sign Language Recognition. In LREC-COLING, pages 5450–5460, 2024. 2, 3

[24] Jie Huang, Wen-Gang Zhou, Houqiang Li, and Weiping Li. Attention-based 3D-CNNs for large-vocabulary sign language recognition. IEEE Transactions on Circuits and Systems for Video Technology, 29(9):2822–2832, 2019. 1, 2, 3

[25] Songyao Jiang, Bin Sun, Lichen Wang, Yue Bai, Kunpeng Li, and Yun Fu. Skeleton Aware Multi-Modal Sign Language Recognition. In CVPR, pages 3413–3423, 2021. 1, 2, 3

[26] Peng Jin, Hongkai Li, Jun Yang, Yazhou Ren, Yuhao Li, Lilan Zhou, Jin Liu, Mei Zhang, Xiaorong Pu, and Siyuan Jing. A large dataset covering the Chinese national sign lan guage for dual-view isolated sign language recognition. Sci entific data, 12(1):660, 2025. 1, 2, 3

[27] Hamid Reza Vaezi Joze and Oscar Koller. MS-ASL: A largescale data set and benchmark for understanding American sign language. In BMVC, page 100, 2018. 3

[28] Mutsuro Kai. Japanese Language 1 (Vol. 1). Mitsumura Tosho, Tokyo, 2015. 4

[29] Alexander Kapitanov, Kvanchiani Karina, Alexander Nagaev, and Petrova Elizaveta. Slovo: Russian sign language dataset. In Lecture Notes in Computer Science, pages 63–73. Springer Nature Switzerland, Cham, 2023. 3

[30] A R Lederberg and V S Everhart. Communication between deaf children and their hearing mothers: the role of language, gesture, and vocalizations. Journal ofspeech, language, and hearing research: JSLHR, 41(4):887–899, 1998. 1

[31] Dongxu Li, Cristian Rodriguez-Opazo, Xin Yu, and Hongdong Li. Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison. In WACV, pages 1448–1458, 2019. 1, 3

[32] A M Martinez, R B Wilbur, R Shay, and A C Kak. Purdue RVL-SLLL ASL database for automatic recognition of American Sign Language. In ICMI, pages 167–172, 2003. 3

[33] Carlos-D Mart´ınez-Hinarejos and Zuzanna Parcheta. Spanish Sign Language Recognition with Different Topology Hidden Markov Models. In Interspeech, pages 3349–3353, 2017. 2, 3

[34] Liliane Momeni, Gul Varol, Samuel Albanie, Triantafyllos¨ Afouras, and Andrew Zisserman. Watch, read and lookup: learning to spot signs from multiple supervisors. In ACCV, pages 291–308, 2020. 3

[35] D F Moores. Educating the Deaf: Psychology, Principles, and Practices. Houghton Mifflin, Boston, 2001. 1

[36] Suneetha Mopidevi, M Prasad, and P Kishore. Multiview meta-metric learning for sign language recognition using triplet loss embeddings. Pattern analysis and applications: PAA, 26(3):1125–1141, 2023. 3

[37] Yuji Nagashima, Daisuke Hara, Yasuo Horiuchi, and Shinji Sako. KoSign: Kogakuin University Japanese Sign Language Multi-Dimensional Database. https://www. nii.ac.jp/dsc/idr/rdata/KoSign/, 2021. Accessed: 2025-10-28. 1, 2, 3

[38] W Neubauer, A Gupta, S Shahryar, M So, D Martin, T Starner, and S Forbes. PopSign v2.0: Extending an Isolated American Sign Language Dataset to Over 360,000 Examples of 562 Concepts. In IEEE FG, 2026. 1, 2, 3, 6, 7

[39] Tamiko Ogura, Tohru Watamaki, and Taichi Inaba. The Japanese MacArthur-Bates communicative development inventories. Nakanishiya Shuppan, Kyoto, 2016. 4

[40] Eng-Jon Ong, H Cooper, N Pugeault, and R Bowden. Sign language recognition using sequential pattern trees. In CVPR, pages 2200–2207, 2012. 3

[41] Yutaka Osugi. A Guide to Hand Gestures and Words (Kitakyushu) (in Japanese). http://www.deafstudies. jp/osugi/kitakyushu/, 2012. Accessed: 2025-11-4. 2, 3

[42] Ilya Ovodov, Petr Surovtsev, Karina Kvanchiani, Alexander Kapitanov, and Alexander Nagaev. Logos as a well-tempered pre-train for sign language recognition. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 24351–24364, Stroudsburg, PA, USA, 2025. Association for Computational Linguistics. 3

[43] Katerina Papadimitriou and Gerasimos Potamianos. Multimodal locally enhanced transformer for continuous sign language recognition. In Interspeech, pages 1513–1517. ISCA, 2023. 2

[44] Franco Ronchetti, Facundo Manuel Quiroga, Cesar Estre-´ bou, Laura Lanzarini, and Alejandro Rosete. LSA64: An Argentinian Sign Language dataset. arXiv:2310.17429 [cs.CV], 2023. 3

[45] Xin Shen, Shaozu Yuan, Hongwei Sheng, Heming Du, and Xin Yu. Auslan-daily: Australian sign language translation for daily communication and news. In NeurIPS, pages 80455–80469, 2023. 3

[46] Xin Shen, Heming Du, Hongwei Sheng, Shuyun Wang, Huiqiang Chen, Huiqiang Chen, Zhuojie Wu, Xiaobiao Du, Jiaying Ying, Ruihan Lu, Qingzheng Xu, and Xin Yu. MM-WLAuslan: Multi-view multi-modal word-Level Australian Sign Language Recognition dataset. In NeurIPS, pages 69700–69715, 2024. 1, 2, 3

[47] Bowen Shi, Wei-Ning Hsu, Kushal Lakhotia, and Abdelrahman Mohamed. Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction. arXiv preprint arXiv:2201. 02184, 2022. 7

[48] Ozge Mercanoglu Sincan and H Keles. AUTSL: A large scale multi-modal Turkish Sign Language dataset and base line methods. IEEE Access, 8:181340–181355, 2020. 1, 3

[49] Advaith Sridhar, Rohith Gandhi Ganesan, Pratyush Kumar, and Mitesh M Khapra. INCLUDE: A large scale dataset for Indian Sign Language Recognition. In MM, pages 1366– 1375, 2020. 3

[50] Thad Starner, Sean Forbes, Matthew So, David Martin, Rohit Sridhar, Gururaj Deshpande, Sam S Sepah, Sahir Shahryar, Khushi Bhardwaj, Tyler Kwok, Daksh Sehgal, Saad Hassan, Bill Neubauer, S Vempala, Alec Tan, Jocelyn Heath, Unnathi Kumar, Priyanka Mosur, Tavenner Hall, Rajandeep Singh, Christopher Cui, Glenn Cameron, Sohier Dane, and Garrett Tanzer. PopSign ASL v1.0: An isolated American sign language dataset collected via smartphones. In NeurIPS, pages 184–196, 2023. 1, 2, 3, 7

[51] Alyssa Takazume, Noriko Yata, and Yoshitsugu Manabe. Japanese Sign Language Recognition Using Human-Shaped Graph Data. SSRN Electronic Journal, 2023. 2, 3

[52] Wataru Takei and Takashi Torigoe. The role of pointing gestures in the acquisition of Japanese sign language. The Japanese Journal of Special Education, 38(6):51–63, 2001. 1, 4

[53] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. Learning Spatiotemporal Features with 3D Convolutional Networks. In ICCV, pages 4489–4497, 2015. 3

[54] Zhen Wang, Dongyuan Li, Renhe Jiang, and Manabu Okumura. Continuous sign language recognition with multiscale spatial-temporal feature enhancement. IEEE Access, 13:5491–5506, 2025. 2

[55] Morteza Zahedi, Daniel Keysers, Thomas Deselaers, and Hermann Ney. Combination of tangent distance and an image distortion model for appearance-based sign language recognition. In DAGM, pages 401–408, 2005. 3

[56] Weichao Zhao, Hezhen Hu, Wengang Zhou, Yunyao Mao, Min Wang, and Houqiang Li. MASA: Motion-aware masked autoencoder with semantic alignment for sign language recognition. IEEE Transactions on Circuits and Systems for Video Technology, 34(11):10793–10804, 2024. 2, 3, 6, 7

[57] Ronglai Zuo and Brian Mak. Local context-aware selfattention for continuous sign language recognition. In In terspeech, pages 4810–4814. ISCA, 2022. 2

[58] Ronglai Zuo, Fangyun Wei, and B Mak. Natural Language-Assisted Sign Language Recognition. In CVPR, pages 14890–14900, 2023. 1, 2, 3, 6, 7

[59] Ogulcan ˘ Ozdemir, Ahmet Alp Kındıro <sup>¨</sup> glu, Necati Cihan ˘ Camgoz, and Lale Akarun. BosphorusSign22k Sign Lan-¨ guage Recognition Dataset. arXiv:2004.01283 [cs.CV], 2020. 3