# A HamNoSys-Guided Dataset and Baselines for Fine-Grained Isolated Handshape Recognition in Sign Language

Ushnish Sarkar<sup>1,2\*</sup>, Suvajit Patra<sup>3†</sup>, Bhaswar Chattopadhyay<sup>1,2†</sup>, Pranab Singha Roy<sup>1†</sup>, Tapas Samanta<sup>1,2†</sup>

<sup>1\*</sup>Computer and Informatics Group, Variable Energy Cyclotron Centre, Kolkata, 700064, India.

<sup>2</sup>Homi Bhabha National Institute, Mumbai, 400094, India.

<sup>3</sup>Ramakrishna Mission Vivekananda Educational and Research Institute, Belur, 711202, India.

\*Corresponding author(s). E-mail(s): u.sarkar@vecc.gov.in; <sup>†</sup>These authors contributed equally to this work.

## Abstract

Purpose: Fine-grained handshape recognition supports computational sign-language transcription, recognition, and translation, but broad, phonetically defined visual inventories with signer-aware evaluation remain limited. This work introduces a handshape dataset grounded in the languageindependent Hamburg Notation System (HamNoSys) and baseline models for handshape recognition evaluated on the same.

Methods: A balanced dataset of 144,000 RGB images was collected from 15 participants for 160 handshape classes defined by the oficial HamNoSys 4 Handshapes Chart. ResNet-18 and ViT-B/16 as appearance-based models were evaluated on this dataset, while a graph convolutional network and XGBoost were evaluated on the hand landmarks of the images from this dataset. Both a classstratified subject-dependent split and a 15-fold leave-one-subject-out (LOSO) protocol were used. The same model families were additionally assessed on LSWH100 and ASL Fingerspelling Dataset A for external context.

Results: The subject-dependent baselines established reproducible reference performance across all four model families, whereas LOSO evaluation showed some reduction when recognition was required to generalise to unseen participants. Additional analysis has been shown to highlight the presence of visually close handshapes that may possibly confuse the standard models .

Conclusion: The documented acquisition, curation, and complementary evaluation protocols provide a reproducible resource for fine-grained isolated-handshape research and for developing more accessible sign-language technologies.

Keywords: Sign language, HamNoSys, handshape recognition, dataset, leave-one-subject-out evaluation

## 1 Introduction

More than 1.5 billion people worldwide experience some degree of hearing loss [1], while a national survey in the United States estimated that approximately 2.8% of adults use a sign language [2]. Accessible computational tools for sign-language transcription and recognition are therefore relevant to a substantial and diverse population. Sign languages are natural languages with their own phonological, morphological, and syntactic structures [3]. At the sublexical level, signs can be analysed through contrastive manual parameters such as handshape, orientation, location, and movement, together with linguistically relevant non-manual components [3, 4]. Among the manual parameters, handshape provides an important source of lexical contrast and is consequently relevant to computational sign-language recognition and translation [5]. Linguistically grounded representation, resource construction, and evaluation have therefore remained central concerns in sign-language technology [6, 7].

The visual–gestural modality presents a fundamental representational dificulty. Video preserves the multidimensional and temporally organised signal, but it does not by itself provide an explicit, discrete, and searchable description of the articulatory form. Glosses can be used to identify lexical sign types or meanings, but diferences in handshape, orientation, location, movement, and signer-specific realisation are not encoded by them. When physical sign form, phonetic variation, phonological contrast, or cross-linguistic similarity is to be investigated, a form-based transcription system is therefore required. Through such a system, observable sublexical properties can be represented consistently, sign variants can be compared, corpus annotations can be searched, and findings can be evaluated across datasets and sign languages [8, 9]. However, no universally accepted sign-language equivalent of the International Phonetic Alphabet has yet been established [9].

Several transcription systems have been developed at diferent levels of descriptive abstraction. Stokoe notation introduced a parameterbased phonological representation of handshape, location, and movement, principally for American Sign Language [4]. The Liddell–Johnson Movement–Hold model subsequently represented signs through temporally ordered movement and hold segments [10]. Prosodic Model Handshape Coding provides a theoretically motivated phonological representation of selected fingers and joint configuration [11], whereas Sign Language Phonetic Annotation provides a more anatomically detailed description of individual fingers, joints, thumb configuration, and articulator contact [9, 12, 13]. SignWriting serves primarily as a graphical or orthographic representation and has also been employed as an intermediate representation in computational translation [14]. These systems difer in purpose, descriptive coverage, and granularity and are therefore not interchangeable [15, 16]. For the present handshape-centred task, HamNoSys ofers greater articulatory detail and broader cross-linguistic applicability than the original Stokoe system, while avoiding the extensive joint-by-joint description required by Sign Language Phonetic Annotation [13, 15]. Unlike a notation restricted to a particular phonological model, it can also be used as a mostly phonetic description of observable manual form [17, 18]. Most importantly for dataset construction, the oficial, non-exhaustive HamNoSys 4 Handshapes Chart provides publicly illustrated reference forms from which a bounded and visually reproducible class inventory can be defined [19]. Reported inconsistencies in the use of HamNoSys annotations further support the use of labels tied directly to fixed chart illustrations rather than unconstrained transcription [20]. Existing visual resources support several related tasks, including fingerspelling recognition, corpus-derived handshape classification, and synthetic handshape recognition. However, their difering objectives leave a need for a real-image benchmark that combines a broad, transcription-defined handshape inventory with systematic evaluation on unseen participants. The supporting comparison with existing resources is provided in Section 2. This need was addressed through the construction of a balanced, HamNoSys-grounded image dataset and its evaluation under complementary subjectdependent and leave-one-subject-out protocols. The class definition, acquisition procedure, and dataset composition are presented in Section 3, while the baseline models and matched-model experiments on external datasets are presented in Section 4.

## 2 Related Work

Fine-grained handshape inventories distinguish selected fingers, joint configuration, thumb behaviour, inter-finger relations, and contact [11, 21, 22]. Segmental descriptions additionally distinguish static articulation from its temporal organisation within a sign [12]. Because neighbouring handshapes may difer in only one of these properties, fine-grained recognition requires a broader inventory than that provided by language-specific fingerspelling alphabets.

Existing visual resources represent several related but distinct tasks. Continuous-sign corpora provide weak or auxiliary handshape labels [23, 24]; LSWH100 contains synthetic images organised using SignWriting-derived categories [25]; and fingerspelling datasets cover restricted language-specific alphabets or digits [26–29]. Other resources represent complete lexical signs [30] or derive hand-pose categories through feature-based clustering [31]. Their diferences in visual data, class inventory, and linguistic scope are summarised in Table 1. Within the reviewed literature, no resource jointly provides an approximately balanced collection of real isolated-hand images, a broad class inventory grounded in a language-independent phonetic notation, and evaluation under both participantoverlapping and participant-independent protocols. This combination defines the specific resource gap addressed in the present study. Ham-NoSys has otherwise been applied primarily at the sign level in corpora, multilingual lexicons, dictionary transcription, cross-linguistic comparison, and automatic motion generation [18, 32–37]. As summarised in Table 2, these applications encode lexical citation forms, corpus units, or motion sequences rather than balanced image collections organised by isolated handshape. The HamNoSys 4 Handshapes Chart is itself an illustrated reference inventory rather than an image dataset; its use for defining the proposed classes is described in Section 3.

More broadly, gesture-recognition research has addressed human–computer interaction, virtual environments, rehabilitation, and signlanguage processing [38]. General-purpose gesture resources, however, are not necessarily organised as linguistically defined handshape inventories.

Complementary representation families have been adopted for handshape and gesture recognition. RGB images have been processed using convolutional and transformer-based architectures [39, 40], while hand landmarks have been represented as anatomical graphs or fixed-length feature vectors [41–44]. Representative baselines from these families are evaluated in the present study rather than an exhaustive set of architectures.

Evaluation design is particularly important for multi-participant data. Participant-overlapping splits measure recognition when the same participants may occur across partitions, whereas participant-independent protocols assess generalisation to unseen individuals [45]. Both settings are therefore reported, with leave-one-subject-out evaluation used for the systematic participantindependent assessment described in Section 4.

## 3 Dataset Construction

This section defines the handshape class inventory and subsequently describes the acquisition protocol, collection software, participants, dataset organisation, and evaluation splits.

## 3.1 HamNoSys Handshape Inventory and Class Definition

HamNoSys Version 2.0 encodes the manual components of a sign through handshape, orientation, location, and movement [17]. Version 4 additionally supports optional non-manual specifications [18]. Handshapes are constructed compositionally from a basic form and modifiers describing finger selection, bending, thumb position, individual-finger configuration, and intermediate forms. Transitions between handshapes are represented as actions rather than separate dynamic handshape primitives [18].

Because HamNoSys does not define a finite handshape inventory, the oficial, non-exhaustive HamNoSys 4 Handshapes Chart was used to establish a bounded and reproducible class set [19]. Each distinct illustrated hand model in Fig. 1 was treated as one class; blank cells and cells containing only symbols or cross-references were excluded. This procedure produced 160 classes.

The illustrations occupy six Selection rows and four Thumb-opposition rows. The two Thumbopposition rows containing no illustrations—Two Fingers (spread), others in fist position and Four Fingers (spread)—were excluded. Table 3 reports the class counts for the ten populated rows, while Table 4 groups the same 160 classes by chart column: 106 Selection classes and 54 Thumbopposition classes. All dataset codes are authordefined and are not oficial HamNoSys symbols.

Table 1 Representative visual handshape and sign-language resources.
<table><tr><td>Resource</td><td>Visual data</td><td>Classes</td><td>Label basis or scope</td></tr><tr><td>Deep Hand [23]</td><td>&gt; 1 million weakly labelled real frames</td><td>60</td><td>Corpus- and lexicon-derived handshapes</td></tr><tr><td>PHOENIX14T-HS [24]</td><td>Continuous-sign videos</td><td>60</td><td>Auxiliary handshape labels associated with DGS glosses</td></tr><tr><td>LSWH100 [25]</td><td>144,000 synthetic images</td><td>100</td><td>SignWriting-derived Libras handshapes</td></tr><tr><td>ASL Fingerspelling Dataset A [27]</td><td>65,774 real RGB images from five users</td><td>24</td><td>Static ASL alphabet</td></tr><tr><td>Other fingerspelling resources [26, 28, 29]</td><td>Static real or augmented images</td><td>24-41</td><td>ASL, JSL, or Danish letters and digits</td></tr><tr><td>LSA64 [30]</td><td>3,200 videos</td><td>64</td><td>Complete Argentinian Sign Language lexical signs</td></tr><tr><td>Kajiyama et al. [31]</td><td>Hand images from sign-language data</td><td>Data- derived</td><td>Finger-shape and palm-orientation clusters</td></tr><tr><td>Proposed dataset</td><td>144,000 real RGB images from 15 participants</td><td>160</td><td>Chart-defined HamNoSys handshapes</td></tr></table>

Table 2 Documented applications of HamNoSys in sign-language resources.
<table><tr><td>Resource</td><td>Sign language(s)</td><td>Scale or primary unit</td><td>Function of HamNoSys</td></tr><tr><td>DGS Corpus and GLex [18, 32, 33]</td><td>German Sign Language</td><td>Corpus utterances and technical lexical signs</td><td>Corpus-linked annotation and citation-form description</td></tr><tr><td>DICTA-SIGN resources [34, 37]</td><td>BSL, DGS, GSL, and LSF</td><td>Approximately 1,000 signs per language</td><td>Citation-form representation and cross-linguistic comparison</td></tr><tr><td>Corpus-based Dictionary of PJM [35]</td><td>Polish Sign Language 3,476 lexical signs</td><td></td><td>Citation-form transcription</td></tr><tr><td>Motion-generation dataset [36]</td><td>Spanish Sign Language</td><td>754 signs and 6,786 videos</td><td>Intermediate representation for automatic motion generation</td></tr></table>

Table 3 Populated chart selections used as dataset categories.
<table><tr><td>Chart section</td><td>Category</td><td>Code</td><td>Classes</td></tr><tr><td>Selection</td><td>Fist</td><td>F</td><td>12</td></tr><tr><td>Selection</td><td>One Finger</td><td>OF</td><td>20</td></tr><tr><td>Selection</td><td>Two Fingers (nonspread)</td><td>TFN</td><td>17</td></tr><tr><td>Selection</td><td>Two Fingers (spread)</td><td>TFS</td><td>19</td></tr><tr><td>Selection</td><td>Flathand (Four Fingers nonspread)</td><td>FFN</td><td>16</td></tr><tr><td>Selection</td><td>Four Fingers (spread)</td><td>FFS</td><td>22</td></tr><tr><td>Thumb opposition</td><td>One Finger, others in fist position</td><td>OFO</td><td>14</td></tr><tr><td>Thumb opposition</td><td>Two Fingers (nonspread), others in fist position</td><td>TFO</td><td>12</td></tr><tr><td>Thumb opposition</td><td>Four Fingers (nonspread)</td><td>FFO</td><td>11</td></tr><tr><td>Thumb opposition</td><td>One Finger, others extended (spread)</td><td>OFOE</td><td>17</td></tr><tr><td></td><td></td><td>Total</td><td></td></tr></table>

The resulting inventory is restricted to the forms illustrated in the non-exhaustive chart and therefore does not cover every handshape expressible in HamNoSys. Extension beyond these 160 classes would require expert definition and validation.

## 3.2 Participants and Ethics

Fifteen university students aged 23–25 years participated in the data collection. Right-hand dominance was reported by fourteen participants and left-hand dominance by one. For each class, participants examined and reproduced the displayed reference handshape. Finger selection, bending, thumb position, and contact were verified against the reference before recording by an operator with sign-language experience. Participants received an honorarium for their time. The applicable ethical oversight and consent procedures are reported in the Statements and Declarations.

<table><tr><td>Selection</td><td colspan="4">Selected Fingers Extended</td><td colspan="4">Selected Fingers Flattened</td><td colspan="6">Selected Fingers Bent</td><td colspan="3">Selected Fingers Hooked</td><td colspan="6">Derivation Examples</td></tr><tr><td>Fist</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>u</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>9</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0</td><td></td><td>ô</td><td></td><td></td><td>10</td><td></td><td>O²\3</td><td></td><td></td><td>0</td><td></td></tr><tr><td>Finger One</td><td></td><td>S</td><td></td><td>G</td><td></td><td></td><td></td><td>α. ;</td><td></td><td></td><td></td><td></td><td>cf. 3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Two Fingers</td><td></td><td>4</td><td></td><td></td><td></td><td></td><td></td><td>cf. §ª</td><td></td><td></td><td></td><td></td><td>cf. 7º3</td><td></td><td></td><td></td><td></td><td></td><td>3</td><td></td><td></td><td></td></tr><tr><td>nonspread</td><td></td><td></td><td></td><td></td><td>1</td><td></td><td>1</td><td></td><td></td><td></td><td></td><td>à</td><td></td><td></td><td></td><td></td><td>2</td><td>dº83</td><td>d3 </td><td></td><td></td><td></td></tr><tr><td>Fingers Two spread</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Flathand (Four</td><td>è</td><td></td><td></td><td></td><td></td><td></td><td></td><td>α. ;</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>è</td><td>d2,</td><td>2</td><td></td><td></td><td></td></tr><tr><td>Fingers nonspread</td><td></td><td></td><td></td><td></td><td></td><td></td><td>ō</td><td></td><td></td><td></td><td></td><td>δ</td><td>d. e</td><td>C</td><td></td><td></td><td>δ</td><td>o5</td><td></td><td></td><td></td><td></td></tr><tr><td>Fou Finedd</td><td></td><td></td><td></td><td></td><td></td><td></td><td>国</td><td></td><td>8</td><td></td><td>è</td><td>à</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>AM</td></tr><tr><td>Thumb Opposition</td><td colspan="4">Fingertip-</td><td colspan="4"></td><td>Fingertip Thumbtip Fingertip Thumbtip Opposition hitchhiker straigh fingers</td><td>Fingertip Interphalan Thumb&#x27;s geal Joint</td><td>Thumbs Fingertip- Metacarpo phalangeal Joint Opposition</td><td colspan="7">gí Derivation Examples</td><td>E</td><td>4</td><td>g2 j3 4 ]5</td><td></td><td></td></tr><tr><td>others in fist One Finger position</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>d. φ</td><td></td><td></td><td></td><td>SYS³</td><td>o!</td><td>o!</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Two Fingers (nonspread), others in fist</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>cf. ¢</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>position Two Finge (spread).</td><td>oº</td><td>=23 ea. à</td><td>=23</td><td>323</td><td>oº </td><td>523</td><td>S23 e.</td><td>5a</td><td>d. ¿</td><td>àa3</td><td>223</td><td>2a3</td><td>S:82</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>others in fist position Four</td><td colspan="4"></td><td colspan="4"></td><td colspan="8"></td><td colspan="7"></td></tr><tr><td>Fingers (nonspread)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>d.9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>M2 </td><td></td><td></td><td>cf. §</td><td></td><td></td><td></td><td>cf. §</td><td></td><td>cf. y</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OnoFi inger. extended (spread)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>cf. §</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>#</td><td></td><td></td><td></td><td></td></tr></table>

Fig. 1 The oficial, non-exhaustive HamNoSys 4 Handshapes Chart [19]. Each distinct drawn hand model is treated as one dataset class.  
Table 4 Number of drawn models under each chart column group.
<table><tr><td>Chart section</td><td>Column group</td><td>Code</td><td>Classes</td></tr><tr><td>Selection</td><td>Selected Fingers Extended</td><td>SFE</td><td>24</td></tr><tr><td>Selection</td><td>Selected Fingers Flattened</td><td>SFF</td><td>17</td></tr><tr><td>Selection</td><td>Selected Fingers Bent</td><td>SFB</td><td>24</td></tr><tr><td>Selection</td><td>Selected Fingers Hooked</td><td>SFH</td><td>21</td></tr><tr><td>Selection</td><td>Derivation Examples</td><td>DE</td><td>20</td></tr><tr><td>Thumb opposition</td><td>Fingertip-Thumbtip Opposition w/fingers rounded</td><td>FTR</td><td>16</td></tr><tr><td>Thumb opposition</td><td>Fingertip-Thumbtip Opposition w/fingers flattened</td><td>FTF</td><td>16</td></tr><tr><td>Thumb opposition</td><td>Fingertip-Thumbtip Opposition w/Hitchhiker&#x27;s fingers</td><td>FTH</td><td>4</td></tr><tr><td>Thumb opposition</td><td>Fingertip Thumb&#x27;s Interphalangeal-joint opposition</td><td>FTI</td><td>4</td></tr><tr><td>Thumb opposition</td><td>Fingertip Thumb&#x27;s Metacarpophalangeal-joint opposition</td><td>FTM</td><td>4</td></tr><tr><td>Thumb opposition</td><td>Other Derivation Examples</td><td>DE</td><td>10</td></tr><tr><td></td><td></td><td>Total</td><td>160</td></tr></table>

## 3.3 Dataset recording

Two interfaces were provided by the acquisition application (Fig. 2). The target HamNoSys handshape and live camera view were displayed on the actor panel, while the reference, incoming stream, and recording controls were displayed on the operator panel. The recording duration was set to 10 seconds.

Recordings were made indoors against a constant dull-white background using a tripodmounted Logitech Brio RGB camera at 30 frames/s and 640 × 480 pixels. The acquisition program was run on an Intel Core i7 Windows workstation, while the session was managed from a separate workstation (Fig. 3).

No instrumented gloves or body-mounted sensors were used. The recording area was kept reasonably uncluttered to limit irrelevant background variation and to support later hand localisation. Participants were free to adjust their upper-body posture and hand position within the camera view. The only required actions were to form the target handshape and perform the prescribed rotation of the hand.

Every class defined in Section 3.1 was performed by each participant. After the referenceguided verification described above, a 10-second clip was recorded while the participant maintained the handshape and slowly rotated the hand about two approximately orthogonal axes. This eficiently introduced viewpoint, apparent overlap, and self-occlusion variation without intentionally changing the class. Each 10-second clip recorded at 30 frames/s yielded 300 RGB frames. To reduce temporal redundancy while retaining samples throughout the hand rotation, every fifth frame was selected, producing 60 images per clip. Participant, HamNoSys class, source-video, frameindex, and dominant-hand metadata were stored for each image, yielding 144,000 labelled images. MediaPipe Hands [41], configured as described in Section 4, localised the metadata-defined dominant hand in 139,199 images. These images formed the common modelling subset for all four baselines, thereby preventing representationdependent sample selection. The remaining 4,801 images were retained in the complete dataset. The pipeline is shown in Fig. 4.

## 3.4 Dataset Organisation and Naming Convention

The dataset is organised hierarchically to preserve the provenance of every image and its correspondence with the chart-defined handshape inventory. The root directory contains 15 participant folders, labelled s0001 to s0015. Each participant folder is divided into the ten populated HamNoSys categories defined in Table 3. Within each category, images are grouped by handshape subclass and chart variation.

A class folder follows the naming convention

## {subclass}\_{category}{variation},

where {subclass} identifies the relevant chart-column group, {category} denotes the dataset-specific selection code, and {variation} indexes the corresponding drawn hand model within that combination. For example, SFE\_OF2 denotes the second Selected Fingers Extended form in the One Finger selection category, whereas DE\_F4 denotes the fourth Derivation Example associated with the Fist category.

Individual image files retain their source-video identifier and frame index using the convention

{video}\_frame\_{frameindex}.png.

Thus, a path such as

$$
\mathtt { s 0 0 0 1 / 0 F / S F E _ { - } 0 F 2 / v 0 0 0 0 0 1 4 _ { - } f r a m e _ { - } 0 0 0 0 0 . p n g }
$$

identifies participant s0001, the One Finger selection category, the second Selected Fingers Extended class, source video v0000014, and extracted frame 00000. Similarly,

$$
\mathbf { s } 0 0 0 1 / \mathrm { F / D E \_ F 4 } / \mathrm { v } 0 0 0 0 1 1 \mathbf { 1 } \mathbf { 1 } \mathbf { - } \mathbf { f r a m e } \_ { 0 } 0 0 0 0 \mathbf { . p n g }
$$

corresponds to the fourth derivation-example class under the Fist selection.

This naming scheme makes each image traceable to its participant, selection category, chartdefined handshape class, source recording, and frame position. A supplementary class-mapping file provides the complete correspondence between the 160 dataset class identifiers and the illustrated hand models in the oficial HamNoSys 4 Handshapes Chart.

![](images/b795eaa7c8d52ce72597e402bb0fa69f93a857c6f53b1a674fb3f2f126af5e7f.jpg)  
Fig. 2 Custom data-acquisition software: (a) actor interface displaying the target handshape and live participant view, and (b) operator/controller interface used for verification and recording control.

![](images/a47c65821a4186dff3d17d2715a8c3bd9a20ab07c8563591322b85ea09c959d6.jpg)  
Fig. 3 Controlled classroom recording environment showing the participant, operator, acquisition computer, and tripodmounted RGB camera from three viewpoints.

## 3.5 Dataset Characteristics and Class Distribution

The complete dataset and common modelling subset are summarised in Table 5. The 144,000 images correspond to 900 per class and 60 per class per participant.

The modelling subset remains approximately uniform (Fig. 5), ranging from 812 images for DE\_F3 to 900 for FTR\_TFO1 and SFB\_TFS3.

Within-class variation in viewpoint, apparent orientation, hand position, and self-occlusion was introduced by the video-based acquisition procedure. Figure 6 presents four frames from each of two representative recordings, with one participant , enacting diferent variations of a handshape, shown per row.

## 3.6 Evaluation Protocols and Data Splits

Two complementary protocols are defined over the 139,199-image modelling subset. A class-stratified frame-level split is used to provide a subjectdependent reference comparable with conventional image-classification benchmarks, including datasets for which signer identities are unavailable. Because participant identity was preserved during collection, the stricter assessment of generalisation to unseen participants is provided by a 15-fold leave-one-subject-out (LOSO) protocol.

Table 5 Principal characteristics of the HamNoSys handshape dataset.
<table><tr><td>Characteristic</td><td>Value</td></tr><tr><td>Total images</td><td>144,000</td></tr><tr><td>Images used for modelling Images excluded from modelling</td><td>139,199 (96.66%) 4,801 (3.34%)</td></tr><tr><td>Handshape classes</td><td>160</td></tr><tr><td>Participants</td><td>15</td></tr><tr><td>Image resolution</td><td>640 × 480 pixels</td></tr><tr><td>Image modality</td><td>RGB</td></tr><tr><td>Mean images per class in total</td><td>900</td></tr><tr><td>subset</td><td></td></tr><tr><td>Mean images per class in mod- elling subset</td><td>869.99</td></tr><tr><td>Minimum images in a class in</td><td>812</td></tr><tr><td>modelling subset Maximum images in a class in</td><td>900</td></tr></table>

![](images/6342788f8be83c01cced67545b647e1e5ca77386432078f53e741b50c74f2f9e.jpg)

![](images/4d431e27087577cd2c088aa8806bdea990a52311ea495d0e6f9ad9e46de31742.jpg)  
Fig. 4 Dataset construction and common modelling-subset selection pipeline.

![](images/194cf619a8d7144c77de1ef3cccee9cc5f9837c4c9eb982fbbeb78a39073067b.jpg)  
Fig. 5 Class balance of the 139,199-image modelling subset. (a) Rank-ordered image counts for the 160 handshape classes, with the minimum and maximum class sizes highlighted. The dashed line indicates the mean class size. (b) Boxplot and histogram summarizing the distribution of per-class image counts. Class sizes range from 812 to 900 images, with a mean of 869.99 images per class.

## 3.6.1 Subject-Dependent Protocol

For the subject-dependent evaluation, images were partitioned at frame level using a classstratified random 70:15:15 split. Temporal redundancy was reduced before this split by retaining every fifth frame, as described in Section 3.3. The resulting counts are reported in Table 6. Because all participants may occur in every partition, recognition under seen-participant conditions is measured by this protocol; unseen-participant generalisation is assessed separately by LOSO.

![](images/2841bd8bcc2404682dd85afcce346d061227d7d1024cc3afd9a69a1ce3b16f9a.jpg)

![](images/255420a2f655a6f63cb8316504c09d4e47589ea13f7cbffc349a6149640c0d80.jpg)

![](images/353764d3c1e7f58a38ab0c305a7bdaf4eddd712ced2f5b0bd663ab7444131c08.jpg)

![](images/4d9de5e13f25c2d61a0118a322a22907062ec05efb2182c41cf9aaf4656836b8.jpg)  
Fig. 6 Representative dataset frames from two participants. Each row contains four frames from one participant, illustrating variation in viewpoint, hand position, articulation, and appearance. Faces are blurred to protect participant identity.

Table 6 Subject-dependent frame-level split of the 139,199-image modelling subset.
<table><tr><td>Partition</td><td>Images</td><td>Percentage</td><td>Approx. per class</td></tr><tr><td>Training</td><td>97,370</td><td>69.95%</td><td>609</td></tr><tr><td>Validation</td><td>20,806</td><td>14.95%</td><td>130</td></tr><tr><td>Test</td><td>21,023</td><td>15.10%</td><td>131</td></tr><tr><td>Total</td><td>139,199</td><td>100%</td><td>870</td></tr></table>

## 3.6.2 Subject-Independent Protocol (LOSO)

Subject-independent performance was evaluated using 15-fold leave-one-subject-out crossvalidation, as summarised in Table 7. In each fold, one participant was reserved exclusively for testing, while images from the remaining 14 participants were divided into class-stratified training and validation partitions using an 85:15 ratio. Identical partitions were used for all four baselines, and no data from the held-out participant were used for model fitting, validation, checkpoint selection, or other training-stage decisions. Each participant served as the test subject once, and performance was reported as the mean and sample standard deviation across the 15 folds.

Table 7 Structure of each leave-one-subject-out fold. Exact image counts vary with the number of usable images contributed by the held-out subject.
<table><tr><td>Component</td><td>Definition</td></tr><tr><td>Test set</td><td>All usable images from one held-out subject</td></tr><tr><td>Development set</td><td>All usable images from the remaining 14 subjects</td></tr><tr><td>Training partition</td><td>85% of the development images, selected using</td></tr><tr><td>Validation partition</td><td>class-stratified sampling 15% of the development images, selected using the same class-stratified split</td></tr><tr><td>Number of folds</td><td>15, with each subject held out exactly once</td></tr><tr><td>Subject overlap</td><td>None between the test subject and the training or validation partitions</td></tr></table>

## 4 Experiments

Baseline recognition performance is established for the proposed dataset under the subjectdependent and leave-one-subject-out protocols defined in Section 3.6. Four models were evaluated: two appearance-based models operating on RGB hand crops and two landmark-based models operating on MediaPipe hand keypoints. The comparisons are intended to characterise the benchmark rather than propose a new recognition architecture.Figure 7 presents the two complementary evaluation protocols and the model variants used under each protocol.

External context is provided by two datasets. LSWH100 [25] contains 144,000 synthetic images in 100 SignWriting-derived classes with predefined training, validation, and test partitions; model output layers were changed to 100 classes and results computed on the predefined test set. ASL Fingerspelling Dataset A [27] contains 65,774 real RGB observations from five users covering 24 static ASL letters (excluding dynamic J and Z). Output layers were changed to 24 classes. A class-stratified 70:15:15 subject-dependent split and five-fold LOSO were used, with 15% of each remaining-user development set reserved for validation where required.

These are matched-model references, not direct rankings of dataset quality: image origin, class definition and count, participant composition, and acquisition protocol difer across datasets.

## 4.1 Training Configurations and Baseline Models

## RGB Input based models.

For the appearance-based models, MediaPipe Hands [41] was applied in static-image mode to identify the participant’s metadata-defined dominant hand. The hand bounding box was expanded by 15 pixels on each side, subject to the image boundaries, and the resulting crop was resized to 224 × 224 pixels.

For ImageNet-pretrained ResNet-18 [39], the classifier was replaced by a 160-class linear head and only the final residual stage and new head were fine-tuned. For ImageNet-pretrained ViT-B/16 [40], the head was replaced by dropout (0.3) and a 160-class linear layer; the final two transformer blocks, encoder normalisation, and head were fine-tuned. Horizontal flipping, rotation up to $1 5 ^ { \circ }$ , colour jitter, and random erasing were used for ResNet-18 augmentation. Afine and perspective transformations and stronger photometric augmentation were additionally used for ViT-B/16. Evaluation images were resized and ImageNet-normalised.

## Landmark Input based models.

MediaPipe Hands was also used to extract 21 dominant-hand landmarks from each image, which were preprocessed as in [43]. Detected left hands were mirrored to a common canonical orientation. The coordinates were translated so that the wrist lay at the origin, rotated into a hand-centred coordinate frame, and scaled by the maximum landmark distance. A joint-angle feature, normalised by $\pi ,$ was computed for the 15 internal finger joints; the wrist and fingertips were assigned an angle of zero. Each node was therefore represented by

$$
( x , y , z , \theta ) ,
$$

giving a $2 1 \times 4$ feature matrix.

The anatomical hand connections were used as edges by the graph convolutional network, with self-connections introduced during graph convolution. Five graph-convolutional layers with output dimensions [512, 480, 448, 416, 352], GELU activations, residual connections, batch normalisation, dropout of 0.1, global mean pooling, and a 160- class linear output layer were included.

An 84-dimensional vector formed from each landmark’s x, $y ,$ and z coordinates and joint angle was used by the XGBoost baseline [44]:

$$
2 1 \times ( x , y , z , \theta ) .
$$

For XGBoost, 1,200 boosting rounds were used with maximum depth 10, minimum child weight $^ { 3 , }$ learning rate 0.02180, gamma 0.01248, subsample ratio 0.6762, column-sampling ratio 0.7482, $\ell _ { 2 }$ regularisation 0.01875, $\ell _ { 1 }$ regularisation 0.09311, and histogram-based tree construction.

## 4.2 Evaluation Measures

The subject-dependent results for the proposed dataset and the matched-model results for LSWH100 and ASL Fingerspelling Dataset A are reported in Table 9. Because each partition may contain frames from all participants, this protocol measures recognition under participant overlap and provides a seen-participant reference. It is not interpreted as an estimate of generalisation to unseen participants.

![](images/cdc5dea3702ca9a1f344701ce6ea18db9c81c1f38beddd461ddbedd98fbc38fe.jpg)  
Fig. 7 Overview of the evaluation design and baseline model variants.

Table 8 Training configuration of the neural baseline models.
<table><tr><td>Setting</td><td>ResNet-18</td><td>ViT-B/16</td><td>GCN</td></tr><tr><td>Pretraining</td><td>ImageNet</td><td>ImageNet</td><td>None</td></tr><tr><td>Optimizer</td><td>Adam</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>6.451 × 10 -4</td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td><td>3.285 × 10−5</td></tr><tr><td>Batch size</td><td>64</td><td>32</td><td>128</td></tr><tr><td>Maximum epochs</td><td>60</td><td>60</td><td>150</td></tr><tr><td>Early-stopping patience</td><td>8</td><td>6</td><td>10</td></tr><tr><td>Learning-rate schedule</td><td>ReduceLROnPlateau</td><td>CosineAnnealingLR</td><td>ReduceLROnPlateau</td></tr><tr><td>Label smoothing</td><td>0.1</td><td>0.1</td><td>None</td></tr><tr><td>Input</td><td>224 × 224 RGB</td><td>224 × 224 RGB</td><td>21 × 4 graph</td></tr></table>

## 4.3 Results

## 4.3.1 Subject-Dependent Performance and External Reference

The subject-dependent results for the proposed dataset and the matched-model LSWH100 and ASL Fingerspelling Dataset A references are reported in Table 9. Because frames from all participants may occur in each partition, these values provide a seen-signer reference rather than an estimate of performance on an unseen signer.

On the proposed 160-class dataset, the highest top-1, top-3, and top-5 accuracies were achieved by ViT-B/16, at 86.20%, 95.99%, and 97.79%, respectively. ResNet-18 achieved a comparable top-1 accuracy of 84.72%. The higher top-1 accuracies of the two RGB-based models relative to the landmark-based baselines are consistent with finegrained distinctions benefiting from appearance information that is not completely retained by the 21-point landmark representation. Among the landmark-based models, GCN exceeded XGBoost by 2.87 percentage points in top-1 accuracy, suggesting an advantage from explicitly representing the anatomical connectivity of the hand.

Table 9 Within-dataset test performance on the proposed HamNoSys dataset and ASL Fingerspelling Dataset A under subject-dependent evaluation, and on LSWH100 using its predefined split (%).
<table><tr><td rowspan="2"></td><td colspan="3">Accuracy</td><td colspan="3">Weighted average</td><td colspan="3">Macro average</td></tr><tr><td>@1</td><td>@3</td><td>@5</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td colspan="10">Model Proposed HamNoSys dataset (160 classes)</td></tr><tr><td>ResNet-18</td><td>84.72</td><td>94.62</td><td>96.78</td><td>85.07</td><td>84.72</td><td>84.74</td><td>85.09</td><td>84.72</td><td>84.75</td></tr><tr><td>ViT-B/16</td><td>86.20</td><td>95.99</td><td>97.79</td><td>86.50</td><td>86.20</td><td>86.19</td><td>86.51</td><td>86.19</td><td>86.20</td></tr><tr><td>GCN</td><td>72.44</td><td>88.44</td><td>92.74</td><td>72.62</td><td>72.44</td><td>72.38</td><td>72.60</td><td>72.41</td><td>72.36</td></tr><tr><td>XGBoost</td><td>69.57</td><td>85.23</td><td>90.06</td><td>69.76</td><td>69.57</td><td>69.55</td><td>69.74</td><td>69.54</td><td>69.53</td></tr><tr><td colspan="10">LSWH100 (100 classes)</td></tr><tr><td>ResNet-18</td><td>86.70</td><td>97.17</td><td>98.67</td><td>87.11</td><td>86.70</td><td>86.69</td><td>87.11</td><td>86.70</td><td>86.69</td></tr><tr><td>ViT-B/16</td><td>81.55</td><td>95.85</td><td>98.00</td><td>82.70</td><td>81.55</td><td>81.53</td><td>82.70</td><td>81.55</td><td>81.53</td></tr><tr><td>GCN</td><td>79.54</td><td>94.82</td><td>96.79</td><td>80.11</td><td>79.54</td><td>79.52</td><td>79.80</td><td>79.54</td><td>79.36</td></tr><tr><td>XGBoost</td><td>72.26</td><td>90.50</td><td>94.76</td><td>72.96</td><td>72.26</td><td>72.29</td><td>72.71</td><td>72.23</td><td>72.14</td></tr><tr><td colspan="10">ASL Fingerspelling Dataset A (24 classes)</td></tr><tr><td>ResNet-18</td><td>99.94</td><td>100.00</td><td>100.00</td><td>99.94</td><td>99.94</td><td>99.94</td><td>99.94</td><td>99.94</td><td>99.94</td></tr><tr><td>ViT-B/16</td><td>99.72</td><td>100.00</td><td>100.00</td><td>99.72</td><td>99.72</td><td>99.72</td><td>99.72</td><td>99.71</td><td>99.71</td></tr><tr><td>GCN</td><td>97.21</td><td>99.09</td><td>99.35</td><td>97.25</td><td>97.21</td><td>97.21</td><td>97.17</td><td>96.98</td><td>97.05</td></tr><tr><td>XGBoost</td><td>97.66</td><td>99.17</td><td>99.45</td><td>97.69</td><td>97.66</td><td>97.67</td><td>97.41</td><td>97.52</td><td>97.46</td></tr></table>

P = precision; R = recall. The external datasets have diferent class inventories and acquisition conditions; their results therefore provide context rather than direct estimates of relative dataset quality.

For every model, the weighted and macro scores were closely aligned. This agreement is consistent with the approximately balanced class distribution and indicates that the aggregate results were not dominated by a small number of larger classes.

The highest LSWH100 top-1 accuracy was obtained by ResNet-18 at 86.70%. Relative to the proposed dataset, the change in top-1 accuracy ranged from −4.65 to +7.10 percentage points and difered across models. This change in model ordering, together with the diferent image origins and 100- versus 160-class inventories, prevents a direct ranking of dataset quality from these within-dataset results.

Top-1 accuracy above 97% was obtained by all four models on ASL Fingerspelling Dataset A. These near-ceiling participant-overlapping results must be interpreted in relation to its smaller 24- class alphabetic inventory and diferent acquisition conditions. The external evaluations therefore establish matched-model reference points rather than direct measures of the relative quality of the three datasets. Because ViT-B/16 achieved the strongest subject-dependent performance on the proposed dataset, it was selected for the subsequent confusion analysis. The broad-category confusion matrix in Figure 8 is strongly concentrated along the diagonal, indicating that the ten broad handshape categories were generally separated successfully. The remaining of-diagonal predictions occurred primarily between related categories and motivated a finer target-level analysis.

As shown in Figure 1, several target classes difer only in subtle properties such as finger selection, bending, thumb position, or contact. The five target-level pairs with the largest bidirectional confusion counts are shown in Figure 9. For each pair, the pair error rate was calculated as the total number of errors in both directions divided by the combined support of the two classes.

![](images/a94981146791543a5b24d572c0302770d116154cbcb77efbb3f4851b33deee32.jpg)  
Fig. 8 Broad-category confusion matrix for ViT-B/16 under subject-dependent evaluation. Rows denote true categories and columns denote predicted categories; cell values are test-sample counts.

## 4.3.2 Leave-One-Subject-Out Performance

The LOSO results for the proposed dataset and ASL Fingerspelling Dataset A are reported in Table 10. LOSO evaluation could not be conducted on LSWH100 because signer-identity information is not provided with that dataset.

On the proposed dataset, numerically close mean top-1 accuracies were obtained by ResNet-18 and ViT-B/16, at 45.38% and 45.22%, respectively. Although its top-1 accuracy was lower,

GCN achieved the highest top-3 and top-5 accuracies, at 69.23% and 78.58%. This result indicates that the landmark graph frequently retained the correct class among its leading predictions even when the top-ranked prediction was incorrect.

Relative to the participant-overlapping evaluation, the ResNet-18 and ViT-B/16 top-1 accuracies decreased by 39.34 and 40.98 percentage points, respectively. The fold standard deviations also demonstrate substantial variation among held-out participants. These results identify unseen-participant generalisation as the principal challenge of the proposed 160-class benchmark rather than indicating a failure of withinparticipant handshape recognition.

![](images/2539bded908fb27c377a09f530934c05c831ce3336992c904a4c672073ca3899.jpg)  
Fig. 9 Five most frequently confused target-level handshape pairs for ViT-B/16 under subject-dependent evaluation. HamNoSys chart illustrations are displayed alongside the corresponding class codes. The left and right bars report directional misclassification counts, while the pair error rate represents the total bidirectional errors relative to the combined support of the two classes.

On ASL Fingerspelling Dataset A, mean LOSO top-1 accuracy ranged from 82.20% for ViT-B/16 to 87.40% for ResNet-18. These values are not directly comparable with those of the proposed dataset because ASL Dataset A contains only 24 classes and was collected under diferent conditions. In contrast, the proposed inventory contains 160 fine-grained classes, including several visually similar handshapes that difer in limited articulatory properties. The confusion pairs in Figure 9 illustrate this fine-grained separation problem. Improved representations of local finger articulation and greater robustness to inter-participant variation are therefore important directions for further modelling.

Per-participant top-1 accuracies for the proposed dataset are shown in Figure 10.

## 5 Conclusion

A balanced handshape dataset grounded in the oficial HamNoSys 4 Handshapes Chart was presented, comprising 144,000 RGB images from 15 participants across 160 illustrated classes. RGB appearance baselines were provided by ResNet-18 and ViT-B/16, while hand-landmark baselines were provided by a GCN and XGBoost. Complementary references for seen-participant and participant-disjoint recognition were provided by the subject-dependent and LOSO protocols.

A substantial efect of evaluation protocol on recognition performance was observed, and the 160-class inventory remained challenging under participant-disjoint testing. Matched-model evaluations were also conducted on synthetic LSWH100 and real ASL Dataset A. The dataset and baselines are intended to support phonologygrounded research and the development of transcription, recognition, and translation tools across sign languages, including under-resourced settings where dictionaries may be available but labelled datasets remain scarce.

## Limitations.

Few limitations should be noted. Data were collected from 15 university students in a controlled indoor environment using a single RGB camera, and broader demographic, environmental, and sensor variability was therefore not represented. The class inventory was restricted to the 160 static, single-hand forms illustrated in the nonexhaustive HamNoSys chart; dynamic transitions, two-handed configurations, orientation, location, movement, and non-manual components were not included. Reference correspondence was verified by an operator with sign-language experience, but further annotation validation by HamNoSys specialists may strengthen the resource. Finally, only four baseline model families were evaluated. More diverse participants, less-controlled acquisition settings, and advanced fine-grained models should be investigated in future work.

Table 10 Leave-one-subject-out performance on the proposed HamNoSys dataset and ASL Fingerspelling Dataset A (%). Values are reported as mean ± sample standard deviation across 15 held-out subjects for the proposed dataset and five held-out users for ASL Dataset A.
<table><tr><td>Model</td><td>Accuracy@1</td><td></td><td>Accuracy@3</td><td>Accuracy@5</td></tr><tr><td colspan="5">Proposed HamNoSys dataset (15 subjects)</td></tr><tr><td>ResNet-18</td><td> $4 5 . 3 8 \pm 7 . 4 8$ </td><td> $6 7 . 7 2 \pm 8 . 8 0$ </td><td></td><td> $7 5 . 7 4 \pm 8 . 3 5$ </td></tr><tr><td>ViT-B/16</td><td> $4 5 . 2 2 \pm 6 . 9 2$ </td><td> $6 8 . 4 7 \pm 7 . 7 5$ </td><td></td><td> $7 6 . 4 9 \pm 7 . 1 8$ </td></tr><tr><td>GCN</td><td> $4 3 . 4 9 \pm 7 . 7 5$ </td><td> $6 9 . 2 3 \pm 9 . 1 5$ </td><td></td><td> $7 8 . 5 8 \pm 8 . 2 5$ </td></tr><tr><td>XGBoost</td><td> $3 9 . 6 6 \pm 6 . 8 7$ </td><td> $6 4 . 2 9 \pm 8 . 7 5$ </td><td></td><td> $7 4 . 2 9 \pm 8 . 2 2$ </td></tr><tr><td colspan="5">ASL Fingerspelling Dataset A (5 users)</td></tr><tr><td>ResNet-18</td><td> $8 7 . 4 0 \pm 4 . 5 0$ </td><td> $9 6 . 3 2 \pm 1 . 2 1$ </td><td></td><td> $9 7 . 9 5 \pm 0 . 6 8$ </td></tr><tr><td>ViT-B/16</td><td> $8 2 . 2 0 \pm 5 . 3 6$ </td><td> $9 5 . 2 3 \pm 1 . 5 6$ </td><td></td><td> $9 7 . 4 4 \pm 0 . 7 3$ </td></tr><tr><td>GCN</td><td> $8 4 . 2 2 \pm 4 . 0 3$ </td><td> $9 5 . 8 6 \pm 1 . 3 1$ </td><td></td><td> $9 7 . 7 5 \pm 0 . 8 1$ </td></tr><tr><td>XGBoost</td><td> $8 6 . 1 4 \pm 3 . 8 1$ </td><td></td><td> $9 6 . 0 0 \pm 0 . 9 6$ </td><td> $9 7 . 7 6 \pm 0 . 4 9$ </td></tr><tr><td colspan="5"></td></tr><tr><td></td><td>Weighted average</td><td> $F _ { 1 }$ </td><td></td><td>Macro average</td><td> $F _ { 1 }$ </td></tr><tr><td colspan="6">Model P R  $H a m N o S y s ~ d a t a s e t$   $( 1 5 ~ s u b j e c t s )$ </td></tr><tr><td>Proposed</td><td></td><td> $4 3 . 9 2 \pm 7 . 5 6$ </td><td></td><td></td><td></td></tr><tr><td>ResNet-18</td><td> $4 6 . 8 0 \pm 7 . 7 7$   $4 5 . 3 8 \pm 7 . 4 8$ </td><td> $4 3 . 6 3 \pm 7 . 0 7$ </td><td> $4 6 . 7 6 \pm 7 . 8 1$   $4 7 . 5 6 \pm 7 . 3 1$ </td><td> $4 5 . 3 9 \pm 7 . 4 7$   $4 5 . 2 2 \pm 6 . 8 9$ </td><td> $4 3 . 9 0 \pm 7 . 5 7$   $4 3 . 5 9 \pm 7 . 0 5$ </td></tr><tr><td>ViT-B/16</td><td> $4 7 . 6 4 \pm 7 . 2 9$ </td><td> $4 5 . 2 2 \pm 6 . 9 2$   $4 3 . 4 9 \pm 7 . 7 5$ </td><td> $4 4 . 1 0 \pm 8 . 0 3$ </td><td> $4 3 . 4 9 \pm 7 . 7 3$ </td><td> $4 2 . 1 1 \pm 7 . 6 2$ </td></tr><tr><td>GCN XGBoost</td><td> $4 4 . 1 5 \pm 8 . 0 0$   $4 0 . 4 4 \pm 6 . 9 7$ </td><td> $4 2 . 1 4 \pm 7 . 6 1$   $3 9 . 6 6 \pm 6 . 8 7$   $3 8 . 4 9 \pm 6 . 7 2$ </td><td> $4 0 . 3 9 \pm 7 . 0 0$ </td><td> $3 9 . 6 7 \pm 6 . 8 6$ </td><td> $3 8 . 4 7 \pm 6 . 7 3$ </td></tr><tr><td colspan="6">ASL Fingerspelling Dataset A (5 users)</td></tr><tr><td>ResNet-18</td><td> $8 8 . 7 5 \pm 3 . 9 5$   $8 7 . 4 0 \pm 4 . 5 0$ </td><td> $8 7 . 0 2 \pm 4 . 5 2$ </td><td> $8 8 . 7 4 \pm 3 . 8 2$ </td><td> $8 7 . 4 3 \pm 4 . 6 8$ </td><td> $8 7 . 0 3 \pm 4 . 5 5$ </td></tr><tr><td> $\mathrm { V i T - B } / 1 6$ </td><td> $8 4 . 4 8 \pm 4 . 7 0$ </td><td> $8 2 . 2 0 \pm 5 . 3 6$   $8 1 . 3 5 \pm 5 . 6 0$ </td><td> $8 4 . 4 5 \pm 4 . 7 0$ </td><td> $8 2 . 0 8 \pm 5 . 6 0$ </td><td> $8 1 . 2 4 \pm 5 . 7 4$ </td></tr><tr><td>GCN</td><td> $8 6 . 1 6 \pm 3 . 3 6$ </td><td> $8 4 . 2 2 \pm 4 . 0 3$ </td><td> $8 3 . 9 7 \pm 3 . 9 4$   $8 3 . 0 0 \pm 6 . 7 5$ </td><td> $8 1 . 5 6 \pm 5 . 3 6$ </td><td> $8 0 . 8 5 \pm 6 . 3 1$ </td></tr><tr><td>XGBoost</td><td> $8 8 . 0 8 \pm 2 . 8 8 $ </td><td> $8 6 . 1 4 \pm 3 . 8 1$   $8 5 . 7 1 \pm 3 . 7 0$ </td><td> $8 4 . 6 4 \pm 6 . 0 5$ </td><td> $8 3 . 9 5 \pm 5 . 2 2$ </td><td> $8 2 . 7 2 \pm 6 . 2 3$ </td></tr><tr><td colspan="6"></td></tr></table>

P = precision; R = recall.

conducted in accordance with the applicable institutional requirements.

Consent to participate. Written informed consent was obtained from all participants before data collection. Participation was voluntary, and the study procedure and intended research use of the recordings were explained before the recording sessions.

## Statements and Declarations

Ethics approval. The study was conducted collaboratively by the Variable Energy Cyclotron Centre and Ramakrishna Mission Vivekananda Educational and Research Institute under the ethical and administrative procedures mutually established by the participating organisations. All procedures involving human participants were

Consent for publication. Written consent was obtained for the publication of participant images. All participant faces shown in this article were anonymised before publication.

Data availability. The dataset will be made available after publication upon reasonable request to the corresponding author at u.sarkar@vecc.gov.in. Access will be subject to the conditions established by the collaborating institutions.

![](images/71e835636f3a61ad340e755a2183b7e2a33a613e6d984ef3263c3cf3b4310cf2.jpg)  
Fig. 10 Top-1 accuracy for each held-out signer under LOSO evaluation. Dashed lines indicate the respective across-signer means.

## References

[1] World Health Organization. Deafness. WHO Facts in Pictures (2024). URL https://ww w.who.int/news-room/facts-in-pictures/de tail/deafness. Published 1 February 2024; accessed 21 July 2026.

[2] Mitchell, R. E. & Young, T. A. How many people use sign language? a national health survey-based estimate. The Journal of Deaf Studies and Deaf Education 28, 1–6 (2023).

[3] Sandler, W. & Lillo-Martin, D. Sign Language and Linguistic Universals (Cambridge University Press, Cambridge, UK, 2006).

[4] Stokoe, W. C. Sign Language Structure: An Outline of the Visual Communication Systems of the American Deaf. No. 8 in Studies in Linguistics: Occasional Papers (Department of Anthropology and Linguistics, University of Bufalo, Bufalo, NY, 1960).

[5] Rastgoo, R., Kiani, K. & Escalera, S. Sign language recognition: A deep survey. Expert Systems with Applications 164, 113794 (2021).

[6] Bragg, D. et al. Sign language recognition, generation, and translation: An interdisciplinary perspective. In Proceedings of the

21st International ACM SIGACCESS Conference on Computers and Accessibility, 16– 31 (Association for Computing Machinery, New York, NY, USA, 2019).

[7] De Coster, M., Shterionov, D., Van Herreweghe, M. & Dambre, J. Machine translation from signed to spoken languages: State of the art and challenges. Universal Access in the Information Society 23, 1305–1331 (2024).

[8] Garcia, B. & Sallandre, M.-A. Transcription systems for sign languages: A sketch of the diferent graphical representations of sign language and their characteristics. In Müller, C. et al. (eds.) Body–Language– Communication: An International Handbook on Multimodality in Human Interaction, 1125–1138 (De Gruyter Mouton, Berlin, Germany, 2013).

[9] Tkachman, O., Hall, K. C., Xavier, A. & Gick, B. Sign language phonetic annotation meets phonological CorpusTools: Towards a sign language toolset for phonetic notation and phonological analysis. Proceedings of the Annual Meetings on Phonology 3 (2016).

[10] Liddell, S. K. & Johnson, R. E. American sign language: The phonological base. Sign Language Studies 195–277 (1989).

[11] Eccarius, P. & Brentari, D. Handshape coding made easier: A theoretically based notation for phonological transcription. Sign Language & Linguistics 11, 69–101 (2008).

[12] Johnson, R. E. & Liddell, S. K. Toward a phonetic representation of signs, i: Sequentiality and contrast. Sign Language Studies 11, 241–274 (2011).

[13] Hall, K. C., Mackie, S., Fry, M. & Tkachman, O. SLPAnnotator: Tools for implementing sign language phonetic annotation. In Proceedings of Interspeech 2017, 2083–2087 (2017).

[14] Jiang, Z., Moryossef, A., Müller, M. & Ebling, S. Machine translation between spoken languages and signed languages represented in SignWriting. In Findings of the Association for Computational Linguistics: EACL 2023, 1706–1724 (Association for Computational Linguistics, Dubrovnik, Croatia, 2023). URL https://aclanthology.o rg/2023.findings-eacl.127/.

[15] Hochgesang, J. A. Using design principles to consider representation of the hand in some notation systems. Sign Language Studies 14, 488–542 (2014).

[16] Dhanjal, A. S. & Singh, W. Comparative analysis of sign language notation systems for indian sign language. In 2019 Second International Conference on Advanced Computational and Communication Paradigms (ICACCP), 1–6 (IEEE, Gangtok, India, 2019).

[17] Prillwitz, S., Leven, R., Zienert, H., Hanke, T. & Henning, J. HamNoSys Version 2.0: Hamburg Notation System for Sign Languages: An Introductory Guide, vol. 5 of International Studies on Sign Language and Communication of the Deaf (Signum, Hamburg, Germany, 1989).

[18] Hanke, T. HamNoSys—representing sign language data in language resources and language processing contexts. In Streiter, O. & Vettori, C. (eds.) Proceedings

of the LREC2004 Workshop on the Representation and Processing of Sign Languages: From SignWriting to Image Processing. Information Techniques and Their Implications for Teaching, Documentation and Communication, 1–6 (European Language Resources Association (ELRA), Lisbon, Portugal, 2004). URL https://www.sign -lang.uni-hamburg.de/lrec/pub/04001.html.

[19] Hanke, T. HamNoSys 4 handshapes chart. DGS-Korpus Project, University of Hamburg (2010). URL https://www.sign-lang.un i-hamburg.de/dgs-korpus/files/inhalt\_pd f/HamNoSys\_Handshapes.pdf. Dated 10 June 2010; drawings by Heiko Zienert, Olga Jeziorski, and Andreas Hanß.

[20] Ferlin, M. et al. Quantifying inconsistencies in the hamburg sign language notation system. Expert Systems with Applications 256, 124911 (2024).

[21] Brentari, D. & Eccarius, P. Handshape contrasts in sign language phonology. In Brentari, D. (ed.) Sign Languages, Cambridge Language Surveys, 284–311 (Cambridge University Press, Cambridge, UK, 2010).

[22] Brentari, D., Coppola, M., Cho, P. W. & Senghas, A. Handshape complexity as a precursor to phonology: Variation, emergence, and acquisition. Language Acquisition 24, 283–306 (2017).

[23] Koller, O., Ney, H. & Bowden, R. Deep hand: How to train a cnn on 1 million hand images when your data is continuous and weakly labelled. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 3793–3802 (IEEE, 2016).

[24] Zhang, X. & Duh, K. Handshape-aware sign language recognition: Extended datasets and exploration of handshape-inclusive methods. In Findings of the Association for Computational Linguistics: EMNLP 2023, 2993–3002 (Association for Computational Linguistics, Singapore, 2023). URL https://aclanthology .org/2023.findings-emnlp.198/.

[25] Lobo-Neto, V. C. & Pedrini, H. LSWH100: A handshape dataset for brazilian sign language (Libras) using SignWriting. Data in Brief 56, 110780 (2024).

[26] DataMunge. Sign language mnist. Kaggle dataset (2017). URL https://www.kaggle.c om/datasets/datamunge/sign-language-m nist. CC0: Public Domain; accessed 21 July 2026.

[27] Pugeault, N. & Bowden, R. Spelling it out: Real-time ASL fingerspelling recognition. In 2011 IEEE International Conference on Computer Vision Workshops (ICCV Workshops), 1114–1119 (IEEE, Barcelona, Spain, 2011).

[28] Hosoe, H., Sako, S. & Kwolek, B. Recognition of jsl finger spelling using convolutional neural networks. In 2017 Fifteenth IAPR International Conference on Machine Vision Applications (MVA), 85–88 (IEEE, Nagoya, Japan, 2017).

[29] Jain, S. ADDSL: Hand gesture detection and sign language recognition on annotated danish sign language. arXiv preprint arXiv:2305.09736 (2023). URL https://arxi v.org/abs/2305.09736.

[30] Ronchetti, F., Quiroga, F., Estrebou, C. A., Lanzarini, L. C. & Rosete, A. LSA64: An argentinian sign language dataset. In Proceedings of the XXII Congreso Argentino de Ciencias de la Computación (CACIC 2016), 794–803 (Red de Universidades con Carreras en Informática (RedUNCI), 2016). URL ht tp://sedici.unlp.edu.ar/handle/10915/5676 4.

[31] Kajiyama, T., Endo, R., Kaneko, H., Sano, M. & Shishikui, Y. Sign language image dataset with a hand pose type attribute. In 2022 IEEE International Symposium on Broadband Multimedia Systems and Broadcasting (BMSB), 1–4 (IEEE, 2022).

[32] Hanke, T., Schulder, M., Konrad, R. & Jahn, E. Extending the public DGS corpus in size and depth. In Proceedings of

the LREC2020 9th Workshop on the Representation and Processing of Sign Languages: Sign Language Resources in the Service of the Language Community, Technological Challenges and Application Perspectives, 75–82 (European Language Resources Association (ELRA), Marseille, France, 2020). URL http s://aclanthology.org/2020.signlang-1.12/.

[33] Konrad, R. et al. (eds.) Fachgebärdenlexikon Gesundheit und Pflege (Signum, Seedorf, Germany, 2007). URL http://www.sign-lan g.uni-hamburg.de/glex.

[34] Matthes, S. et al. DICTA-SIGN—building a multilingual sign language corpus. In Proceedings of the LREC2012 5th Workshop on the Representation and Processing of Sign Languages: Interactions between Corpus and Lexicon, 117–122 (European Language Resources Association (ELRA), Istanbul, Turkey, 2012). URL https://www.sign-l ang.uni-hamburg.de/lrec/pub/12016.html.

[35] Łacheta, J., Czajkowska-Kisil, M., Linde-Usiekniewicz, J. & Rutkowski, P. (eds.) Korpusowy Słownik Polskiego Języka Migowego/Corpus-based Dictionary of Polish Sign Language (Faculty of Polish Studies, University of Warsaw, Warsaw, Poland, 2016). URL https://www.slownikpjm.uw.edu.pl/en.

[36] Villa-Monedero, M., Gil-Martín, M., Sáez-Trigueros, D., Pomirski, A. & San-Segundo, R. Sign language dataset for automatic motion generation. Journal ofImaging 9, 262 (2023).

[37] Varanasi, A. B., Sinha, M. & Dasgupta, T. Cross-linguistic phonological similarity analysis in sign languages using HamNoSys. In Proceedings of the Workshop on Sign Language Processing (WSLP), 51–66 (Association for Computational Linguistics, IIT Bombay, Mumbai, India, 2025). URL https: //aclanthology.org/2025.wslp-main.9/.

[38] Mitra, S. & Acharya, T. Gesture recognition: A survey. IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews) 37, 311–324 (2007).

[39] He, K., Zhang, X., Ren, S. & Sun, J. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778 (IEEE, 2016).

[40] Dosovitskiy, A. et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (2021). URL https: //openreview.net/forum?id=YicbFdNTTy.

[41] Zhang, F. et al. MediaPipe Hands: On-device real-time hand tracking. arXiv preprint arXiv:2006.10214 (2020). URL https://arxi v.org/abs/2006.10214.

[42] Kipf, T. N. & Welling, M. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations (2017). URL https://op enreview.net/forum?id=SJU4ayYgl.

[43] Sarkar, U., Chakraborti, A., Samanta, T., Pal, S. & Das, A. Enhancing asl recognition with gcns and successive residual connections. In Palaiahnakote, S. et al. (eds.) Pattern Recognition. ICPR 2024 International Workshops and Challenges, vol. 15616 of Lecture Notes in Computer Science, 3–16 (Springer Nature Switzerland, Cham, 2025).

[44] Chen, T. & Guestrin, C. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794 (Association for Computing Machinery, New York, NY, USA, 2016).

[45] Aly, S. & Aly, W. DeepArSLR: A novel signer-independent deep learning framework for isolated arabic sign language gestures recognition. IEEE Access 8, 83199–83212 (2020).