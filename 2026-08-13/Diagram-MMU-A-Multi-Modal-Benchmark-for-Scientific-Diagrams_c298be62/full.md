# Diagram-MMU: A Multi-Modal Benchmark for Scientific Diagrams

Weihao Bo<sup>1,2∗</sup> Shan Zhang<sup>3†</sup> Yanpeng Sun<sup>4∗</sup> Jie Liu<sup>2,5</sup> Yongke Yao<sup>2,6</sup> Jinhao Du<sup>7</sup> Wei He<sup>2</sup> Kai Zou<sup>8</sup> Zechao Li<sup>1†</sup> Jingdong Wang<sup>2‡</sup>

<sup>1</sup>Nanjing University of Science and Technology <sup>2</sup>Baidu Inc <sup>3</sup>AIML, Adelaide University <sup>4</sup>SUTD <sup>5</sup>Southeast University <sup>6</sup>East China Normal University <sup>7</sup>University of Oxford <sup>8</sup>NetMind.ai

## Abstract

Multimodal Large Language Models (MLLMs) have been growing the capability for scientific writing and collaboration. For example, OpenAI Prism is a free workspace for scientific writing and collaboration. One important feature in Prism is turning scientific diagrams directly into LAT X TikZ code. In this paper, we build a benchmark, Diagram-MMU, a multi-modal benchmark designed to assess MLLMs’ ability for scientific diagram parsing and understanding. Diagram-MMU features 3.7k curated diagrams and 18.3k human-validated questions across six domains. It evaluates MLLMs on three tasks common in vibe writing workspaces: diagram-to-code parsing, diagram-to-code editing, and diagram question answering, alongside agentic settings per task. The evaluation of 12 MLLMs reveals that diagram-to-code tasks are more challenging than diagram question answering: models can reason well over diagrams but struggle to parse and edit them, underscoring the need for methods to enhance MLLMs’ capability in diagram-to-code generation. Under agentic settings, most models improve parsing and editing performance but degrade on question answering, while Claude-4.6 Opus consistently improves across all three tasks.

Date: August 13, 2026 Connection Email: shan.zhang@adelaide.edu.au, yanpeng\_sun@sutd.edu.sg Project Page: https://vi-ocean.github.io/projects/diagram-mmu

![](images/0ca1595b94792ab332105dd795d91b64085530703fd4dd2487038d5c9891eccc.jpg)

## 1 Introduction

In the wake of significant advancements in large language models (LLMs) [1–3], multimodal large language models (MLLMs) [4–7] have rapidly evolved to extend their versatility across vision–language tasks [8–11]. To bring these capabilities into everyday research workflows, OpenAI Prism [12] provides a free vibe writing workspace where any researcher who writes in LAT<sub>E</sub>X can draft papers, generate figures, and convert scientific diagrams into TikZ code.

Scientific diagrams are the primary visual medium through which researchers convey experimental results, mathematical relationships, molecular structures, and circuit topologies [13–15]. A researcher working with diagrams needs to: (1) parse a diagram into editable TikZ code so it can be included in a manuscript; (2) edit a diagram–changing colors, labels, data values, or layout–to fit specific requirements; and (3) describe or reason about diagram to interpret domain-specific semantics of symbols, or even perform hypothetical reasoning (what-if) about how answers change conditioned on modified diagrams.

Table 1 Comparison with diagram-to-code and question-answering benchmarks. #Diag.: unique diagrams across diagram types (#Diag. Type). Tasks include three fundamental tasks (evaluating how to think): Diagram-to-Code Parsing (D2C-P), Diagram-to-Code Editing (D2C-E), and Diagram Question Answering (DQA), alongside agentic settings (evaluating how to act). Evaluation spans multi-levels: object-level F1 for fine-grained element grounding, CrystalBLEU for TikZ syntax correctness, image-level for visual appearance similarity, and answer accuracy (Acc.) for DQA. Benchmarks without TikZ Code (✗) target Python charts, HTML WebUIs, SVG graphics, or no code (ChartE<sup>3</sup>). ✓ is partially supported (limited editing or text-only input). ⊘ denotes the synthetic diagrams.
<table><tr><td></td><td colspan="3">Data</td><td colspan="4">Tasks</td><td colspan="4">Evaluation Metrics</td></tr><tr><td>Benchmark</td><td>TikZ Code</td><td>#Diag. Type</td><td>#Diag.</td><td>D2C-P</td><td>D2C-E</td><td>DQA</td><td>Agentic</td><td>Object</td><td>CBLEU</td><td>Image</td><td>Answer Acc.</td></tr><tr><td>Diagram-to-Code Parsing</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ChartMimic ∅ [14]</td><td>X</td><td>1</td><td>2.4k</td><td>V</td><td>X</td><td>x</td><td>X</td><td>L</td><td>√</td><td>√</td><td>x</td></tr><tr><td>InfiBench-V [16]</td><td>X</td><td>2</td><td>0.3k</td><td>L</td><td>X</td><td>x</td><td>x</td><td>x</td><td>√</td><td>√</td><td>x</td></tr><tr><td>Plot2Code [17]</td><td>X</td><td>1</td><td>0.1k</td><td></td><td>X</td><td>x</td><td>X</td><td>X</td><td>√</td><td>√</td><td>x</td></tr><tr><td>SVG-Diagrams [18]</td><td>X</td><td>4</td><td>0.5k</td><td>√</td><td>X</td><td>X</td><td>x</td><td>X</td><td>X</td><td>√</td><td>x</td></tr><tr><td>DiagramGenBenchmark [19]</td><td>√</td><td>2</td><td>0.4k</td><td>L</td><td>√</td><td>X</td><td>X</td><td>X</td><td>√</td><td>√</td><td>x</td></tr><tr><td>AutomaTikZ [20]</td><td>√</td><td>4</td><td>1.0k</td><td>√</td><td>X</td><td>X</td><td>X</td><td>x</td><td>V</td><td>√</td><td>x</td></tr><tr><td>DeTikZify [21]</td><td>√</td><td>4</td><td>1.5k</td><td>√</td><td>x</td><td>X</td><td>X</td><td>X</td><td>√</td><td>√</td><td>x</td></tr><tr><td>Image2Struct [22]</td><td>√</td><td>4</td><td>0.3k</td><td>√</td><td>x</td><td>X</td><td>X</td><td>X</td><td>X</td><td>√</td><td>x</td></tr><tr><td>Diagram-to-Code Editing</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ChartE3 [23]</td><td>X</td><td>1</td><td>0.8k</td><td>X</td><td>J</td><td>x</td><td>x</td><td>J</td><td>X</td><td>√</td><td>x</td></tr><tr><td>ChartM³∅ [24]</td><td>x</td><td>1</td><td>1.0k</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td><td>√</td><td>√</td><td>x</td></tr><tr><td>Diagram Question Answering</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CharXiv [13]</td><td>x</td><td>1</td><td>2.3k</td><td>X</td><td>X</td><td>√</td><td>X</td><td>x</td><td>X</td><td>X</td><td>√</td></tr><tr><td>ChartQA [25]</td><td>x</td><td>1</td><td>1.0k</td><td>X</td><td>X</td><td>√</td><td>x</td><td>X</td><td>X</td><td>X</td><td>L</td></tr><tr><td>SGP-Bench ∅ [26]</td><td>X</td><td>2</td><td>3.5k</td><td>X</td><td>x</td><td>√</td><td>x</td><td>X</td><td>X</td><td>X</td><td>√</td></tr><tr><td>MATHEMETRIC ∅ [15]</td><td>X</td><td>3</td><td>1.2k</td><td>X</td><td>X</td><td>√</td><td>x</td><td>√</td><td>X</td><td>X</td><td>√</td></tr><tr><td>Diagram-MMU (Ours)</td><td>L</td><td>6</td><td>3.7k</td><td>¬</td><td>L</td><td>一</td><td>√</td><td>J</td><td>1</td><td>L</td><td>L</td></tr></table>

To assist with the above tasks, MLLMs need strong foundational abilities in perception, coding, domain knowledge, and reasoning (e.g., how to think [7]). Moreover, in the vibe writing workspace, models also need to know how to act (agentic ability [7]): searching TikZ for unfamiliar syntax, coding based on the provided visual objects, building on TikZ codes to apply edits, or deciding which of these steps to invoke for task solving. Most existing benchmarks (Table 1) cover narrow diagram types (predominantly charts) with Python or SVG as the code representation, include a single task (coding, editing, or question answering), and no current benchmark designs agentic settings during evaluation.

We introduce Diagram-MMU, a multi-modal benchmark designed to assess MLLMs’ ability across diagram-tocode parsing (D2C-P), diagram-to-code editing (D2C-E), and diagram question answering (DQA), alongside agentic evaluation settings. This benchmark comprises a carefully curated dataset sourced from TikZ/LAT<sub>E</sub>X documentations, with a selection and filtering process to control quality. The final 3,744 unique diagrams paired with 18,305 evaluation samples are cross-validated by 13 graduate students. Specifically, (1) we adopt TikZ code, which is commonly used in paper writing in LAT<sub>E</sub>X and integrates directly into Overleaf and Prism, and cover six scientific diagram types including charts, planar geometry, 3D shapes, graphs, chemistry, and circuit diagrams (Fig. 1)–for the first time covering chemistry and circuits for diagram-to-code tasks; (2) for evaluation metrics on diagram-to-code tasks, we use three levels: image-level measuring visual appearance similarity, code-level measuring syntactic correctness, and our object-level F1 scores measuring the basic objects that the code draws; (3) we design 16 controllable evaluation settings (Table 4): 3 for foundationa ability and 13 for agentic ability–context utilization, tool use, state management, and planning. Importantly, for tool use we build our TikZ search tool as an MCP server [27], enabling MLLMs to selectively access relevant references, reducing the noise of web search and the context rot caused by loading full PDF manuals.

![](images/4e52585050e0503a307f21ac9a55700ddceef2e53f9f091132c8304f30155187.jpg)  
Figure 1 Demo illustration of the six types of diagrams in Diagram-MMU.

Overall, Diagram-MMU is the first benchmark to test both foundational and agentic abilities for solving scientific diagram-related tasks. Evaluation on 12 MLLMs (6 closed-source, 6 open-source) reveals the key findings on foundational ability: (1) Models can reason well over diagrams (DQA accuracy up to 86%) but struggle to code them (D2C-P object-level F1 ranges 31–57%), revealing perception and coding limitations; (2) Models that fail on diagram-to-code parsing also struggle with editing (D2C-E) and answering tasks (DQA); (3) Gemini-3.0 Pro achieves the most balanced profile across three tasks. On the agentic ability, (4) agency benefits editing more than parsing, likely because textual editing instructions help guide tool and context use; (5) most models degrade from excessive retrieval or poorly targeted queries in the TikZ search process, while Claude-4.6 Opus has the strongest tool use ability; (6) planning is the weakest agentic capability, especially on DQA (−0.3 to −8.8 accuracy degradation). Diagram-MMU serves as a pilot evaluation testbed for scientific diagram parsing, editing, and understanding in vibe writing workspaces, and we hope it inspires further works.

## 2 Related Works

Most existing benchmarks focus on charts [13, 17, 23–25] or narrowly cover scientific diagram types [15, 18, 20, 21], with several using synthetic diagrams [15, 24, 26]. Moreover, benchmarks that adopt Python or SVG as the code representation produce standalone scripts or markup outside the LAT<sub>E</sub>X ecosystem, limiting direct integration into scientific authoring environments. We compare Diagram-MMU with existing diagram benchmarks in Table 1; extended discussion is provided in Appendix §A.

Code representation. Most coding benchmarks adopt Python, primarily for charts [14, 17, 23, 24], or SVG/HTML, for icons, fonts, emojis and web UI [16, 18]. Although these languages are common, they are not native to LAT X: they are rendered by matplotlib/CairoSVG into standalone image files rather than compiled inline within LAT<sub>E</sub>X authoring environments (Overleaf). Moreover, those are vector graphics, generating low-level path elements and fail to maintain accurate geometric relations or only generate outputs of limited

![](images/84d2d364b782a4510573032bbd845948b6e3bacd9a7ad2386fbd9e50f3c8481c.jpg)  
Figure 2 Task construction pipeline for Diagram-MMU.

Table 2 Data statistics of Diagram-MMU. In Table 2a, TikZ.p indicates TikZ package; basic graphical objects are shared across diagrams, while domain-specific objects are summarized in the footnote; #Q denotes total questions; Knowledge is domain-specific for the DQA task. Figure 2b is the metadata breakdown of D2C-E, editing dimensions acorss text, color, scope, and layout: scope operates on local transformations and layout operates on global transformations (e.g., chart type conversion).
<table><tr><td rowspan="2">Domain</td><td colspan="2">D2C-P</td><td colspan="2">DQA</td></tr><tr><td>#Q</td><td>TikZ.p</td><td>#Q</td><td>Knowledge</td></tr><tr><td>Charts</td><td>959</td><td>pgfplots</td><td></td><td>[1,834statistics, trend/extremum analysis</td></tr><tr><td>Planar Geom.</td><td></td><td>598 tikz/tkz-euclide</td><td>1,118</td><td>Euclidean notation/formulas</td></tr><tr><td>3D Shapes</td><td>236</td><td>pgfplots</td><td>455</td><td>solid geometry, cross-sections</td></tr><tr><td>Graph</td><td>1,356</td><td>tikz</td><td>2,679</td><td>degree, connectivity, paths, tree</td></tr><tr><td>Chemistry</td><td>187</td><td>chemfig</td><td>366</td><td>bond, atom, functional groups</td></tr><tr><td>Circuit</td><td>403</td><td>circuitikz</td><td>694</td><td>topology, electrical/power laws</td></tr></table>

Shared objects: path:open, path:closed, node:rectangle, node:circle, text, · · · ;  
Domain-specific: data\_point/axis (Charts, 3D), node:vsource/node:resistor (Circuit).

(a)

![](images/939ee9fdca7c22154e343239af47340999525c46f1abcbfadc858d41cb12267b.jpg)  
(b) (Updated)

complexity [20]. Due to the expressiveness and diverse packages supporting science in TikZ, [20–22] build TikZ-based benchmarks, but evaluate only diagram-to-code parsing on limited diagram types. Vibe writing workspaces, e.g., Prism [12], further advance the utilization of TikZ code: even researchers are not familiar with TikZ syntax, vibe writing frameworks can assist them in diagram-to-code parsing, editing, and integrating the output automatically into manuscripts. We thus use TikZ as code representation. Beyond covering broader diagram types and evaluation metrics, we also build a TikZ search tool as an MCP server to provide TikZ syntax references to advance TikZ integration into vibe writing frameworks.

Task coverage and question types. ChartE<sup>3</sup> [23] designs local and global editing templates for charts, and CharXiv [13] introduces descriptive and reasoning question types for chart understanding; Diagram-MMU extends such designs from charts to all six scientific domains. In Diagram-MMU, each diagram is paired with coding, editing, and understanding questions for joint evaluation, and DQA answers are open-ended and graded by an LLM judge, avoiding the selection bias of multiple-choice [28, 29] or yes/no formats.

## 3 Diagram-MMU

## 3.1 Data Collection

Source. As shown in the first column of Fig. 2. We collect the TikZ source code from two primary sources. One source is oficial package handbooks, i.e., the TikZ/PGF manuals: PGFPlots, CircuiTikZ, TKZ-Euclide, ChemFig, and TikZ-Network documentation. The other source is community resources, including texample.net, TeX Stack Exchange, and GitHub tikz\_favorites. There are totally 6,849 pieces of TikZ code, 3,540 and 3,309 per source.

Simplification, filtering, and categorization. We simplify the TikZ code by removing redundant and useless codes, such as preamble packages unrelated to diagrams (e.g., theorem environments). We check the correctness of the simplied code by two ways: compiling the simplified code using pdflatex, lualatex, or xelatex and comparing the rendered images of the original and simplified code. We compare the renderd images by an automatic MLLM model comparison and a further mannual comparison. If the compile is not successful or the rendered images are not the same, the corresponding code sample is removed.

We then manually remove (near-)duplicate and trivial diagrams (e.g., single arrows, borders, or logos). We use the MLLM, Qwen3-VL-235B-A22B, to classify the diagram and the code into 6 domains: charts, planar geometry, 3D shapes, graph structures, chemical expressions, and circuit. Then, we mannually check the classification correctness and make adjustment if necessary. Finally, we have 3,744 diagrams. Fig. 1 presents the examples of the diagrams of the 6 domains. The statistics is given in Table 2.

![](images/0d776b2af802a70d6285d5ef68a44acf1f731a9bd954c295ed3d0eaa5521775d.jpg)  
Figure 3 Task overview of Diagram-MMU. Diagram-to-Code Parsing is to prompt the MLLMs to parse the input diagram into the TikZ code; Diagram-to-Code Editing is to prompt the MLLMs to enerate the TiKZ code that can produce an rendered diagram meeting with the modification requirement; and Diagram Question Answering is to prompt the MLLMs to answer a question about the input diagram.

## 3.2 Task Definition

Diagram-to-Code Pasing (D2C-P). The task is to prompt the MLLM to parse the input diagram into the TikZ code. An example of the prompt is “Recreate this diagram using LaTeX. Provide the full, executable code. Start with \documentclass[tikz,border=5pt]{standalone}\usepackage{pgfplots}... ”. It contains instruction conditioned on a provided preamble (document class, required packages, and libraries). The top row of Fig. 3 illustrates this task.

Diagram-to-Code Editing (D2C-E). This task is to prompt the MLLM to generate the TiKZ code that can produce the rendered diagram meeting with the modification requirement. An example of the prompt is “Convert the diagram in the image into a complete, executable LaTeX code block, applying the following modification: convert the line chart to a scatter plot by removing the lines connecting the data markers. Start with \documentclass[tikz,border=5pt]{standalone}\usepackage{pgfplots}...” The second row of Fig. 3 gives an example.

Diagram question answering (DQA). This task is to prompt the MLLM to answer a question about the input diagram. An example of the prompt is “Which tester category has the maximum value across both the OldSystem and New System datasets?” and the expected answer is “T4”. The third row of Fig. 3 illustrates this task.

## 3.3 Task Construction

Overview. For D2C-P, we use fixed task templates with placeholders instantiated directly from the original diagrams (the first row of Fig. 3). For D2C-E and DQA task generation, we design an agentic pipeline, as shown in the third column of Fig. 2. Given diagrams, TikZ source codes, and task-specific templates as inputs, an agent generates corresponding tasks.

The generation pipeline for D2C-E includes: (1) Template predefinition for each domain; (2) Given diagram, agent select two tasks from the templates of the corresponding domain; (3) Generate the answer from models; (4) Agent-check and modification; (5) Human-check and modification. The DQA task generation pipeline is similar to that of D2C-E; only task templates difer.

Table 3 Definition and examples of D2C-Editing Task in Diagram-MMU.
<table><tr><td>Domain</td><td>Editing Dim.</td><td>Definition</td><td>Example</td></tr><tr><td rowspan="4">Charts</td><td>Text</td><td>Modify labels, tick marks, or legend text.</td><td>Rename label speed to velocity.</td></tr><tr><td>Color</td><td>Change fill, stroke, or text color of elements.</td><td>Change bar color to blue/legend text to red.</td></tr><tr><td>Scope</td><td>Add or remove data series or elements.</td><td>Move the legent to the top-left.</td></tr><tr><td>Layout</td><td>Modify chart type or filter data.</td><td>Convert bar chart to line chart/remove values below 10.</td></tr><tr><td rowspan="4">Planar Geom.</td><td>Text</td><td>Modify point labels or annotations.</td><td>Change vertices from ABC to DEF.</td></tr><tr><td>Color</td><td>Change line or region color.</td><td>Change the line AB to red.</td></tr><tr><td>Scope</td><td>Add, delete, or rotate geometric elements.</td><td>Add midpoint M on segment AB.</td></tr><tr><td>Layout</td><td>Add construction lines or auxiliary geometry.</td><td>Construct the circumcircle of triangle ABC.</td></tr><tr><td rowspan="4">3D Shapes</td><td>Text</td><td>Modify labels of vertices or faces.</td><td>Rename edge AB to MN.</td></tr><tr><td>Color</td><td>Change the color of edges or regions.</td><td>Highlight the cross-section polygon in green.</td></tr><tr><td>Scope</td><td>Add/delete solid elements.</td><td>Add the altitude from A to plane BCD.</td></tr><tr><td>Layout</td><td></td><td></td></tr><tr><td rowspan="4">Graph</td><td>Text</td><td>Modify node or edge labels.</td><td>Rename node A to Start.</td></tr><tr><td>Color</td><td>Change node or edge color.</td><td>Highlight shortest path edges in red.</td></tr><tr><td>Scope</td><td>Add or delete nodes or edges.</td><td>Add an edge between node B and C.</td></tr><tr><td>Layout</td><td>Rearrange layout or extract subgraph.</td><td>Re-layout graph using circular layout.</td></tr><tr><td rowspan="4">Chemistry</td><td>Text</td><td>Modify atom labels, charges, or subscripts.</td><td>Change CH3 to CH2.</td></tr><tr><td>Color</td><td>Change atom or bond color.</td><td>Highlight oxygen atoms in red.</td></tr><tr><td>Scope</td><td>Add or delete bonds or atoms.</td><td>Add a double bond between C and O.</td></tr><tr><td>Layout</td><td></td><td></td></tr><tr><td rowspan="4">Circuit</td><td>Text</td><td>Modify component labels or values.</td><td>Rename the label from V in to V Source.</td></tr><tr><td>Color</td><td>Change wire or component color.</td><td>Highlight power supply line in red.</td></tr><tr><td>Scope</td><td>Swap or reposition components.</td><td>Move capacitor C1 to the right of R2</td></tr><tr><td>Layout</td><td>Modify circuit topology or branch structure.</td><td>Convert a series RLC circuit to a parallel RLC circuit.</td></tr></table>

We provide the detailed agentic pipeline in Appendix §B.2, and all templates in Appendix §B.4 & §B.6. To guarantee quality, we conduct a rigorous manual review: all samples are manually reviewed by 13 graduate students, where each annotator’s assigned samples are cross-validated by another annotator to ensure that the language is clear and unambiguous, and that the question is answerable with logically consistent options and a correct designated answer.

Question types. D2C-P has one standard question type following its task definition: parsing the diagram into the corresponding TikZ code. D2C-E also follows a standard editing format: prompting models to generate the corresponding TikZ code conditioned on the changes, but we cover four editing dimensions, e.g., text, color, scope, and layout. Example illustrations are shown in Table 3.

We construct two types of questions in DQA: descriptive and reasoning, with a total of 60 templates across the 6 domains (see Appendix §B.6 for details). Descriptive questions assess the model’s capability in extracting and aggregating basic information from diagrams, where the correct answer must strictly adhere to domain-specific knowledge, e.g., two nodes connected by a line represent a geometric segment in planar geometry but a single bond between atoms in chemistry. Reasoning questions evaluate the model’s ability to perform numerical computation, where the correct answer requires applying domain-specific formulas and laws, e.g., Ohm’s law for circuit analysis, Euclidean formulas for geometric area computation, or solid geometry formulas for polyhedron volume.

Descriptive questions, constructed from 23 templates, require: (1) identifying symbols and markers per domain (e.g., zigzag symbols as resistors in circuits, bond types and functional groups in chemistry, right-angle squares in geometry); (2) aggregating diagram information to count elements (e.g., vertices in geometry, ticks/legends in charts, atoms/functional groups in chemistry) and to recognize typology and data patterns (e.g., node degree in graphs, increase/decrease trends in charts). See Appendix §B.7 for data illustrations.

Reasoning questions include standard questions (formed by 18 templates) and what-if questions (formed by 19 templates). The what-if questions require the model to predict answer conditioned on a specific element modification, e.g., “if the data value changes to X, what is the mean of the data series?” All questions require: (1) numerical computation by applying domain-specific formulas (e.g., min, max, and median in charts; area and perimeter in geometry); (2) visual reasoning over diagram (e.g., path length and height of tree in graphs; molecular properties from bond structure in chemistry). Our diagram QAs require only domain semantics/laws and numerical computation, rather than advanced expert-level knowledge [30, 31] (see Appendix §B.8 for per-domain examples).

## 3.4 Evaluation Metrics

Diagram-to-Code Parsing and Editing. We evaluation the diagram-to-code parsing and editing tasks from the three aspects: image, code, and object.

Image-based. We compile both generated and ground-truth TikZ code into images and compare them with four classical metrics, following papers [22, 23]:

(1) SSIM [32] compares brightness, contrast, structure patterns (higher is better);

(2) CLIP Score [33] measures semantic similarity between two images in a shared vision-language space (higher is better);

(3) LPIPS [34] uses deep features from a VGG network to measure perceptual distance in a way that matches human judgement (lower is better);

(4) FID [35] compares the distribution of generated and original images using inception network features (lower is better).

Together, these metrics assess whether the rendered output looks visually similar to the input diagram.

Code-based. We use CrystalBLEU [36] to measure the similarity between generated and ground-truth TikZ code, as in [20]. This focuses on how well the model reproduces the correct coding syntax and structure.

Object-based. We compile both generated and ground-truth TikZ code into DVI and convert them to SVG, from which we extract a Semantic Object Model (SOM)—a structured list of typed, attributed elements (e.g., nodes, paths, text labels, data series). We then organize them into type, text, color, and BBox (see Appendix §C.1 for detailed extraction process). Finally, we compute four F1 scores between the generated and ground-truth SOMs, as follows:

(1) $F 1 _ { t y p e }$ measures whether the generated diagram contains the correct types of elements (e.g., node:circle and node:rectangle);

(2) $\mathrm { F } 1 _ { t e x t }$ evaluates exact string matching of text labels and numeric values;

(3) $F 1 _ { c o l o r }$ compares element colors via permutation-based assignment using CIEDE2000 perceptual distance [14];

(4) $F 1 _ { b b o x }$ assesses spatial positioning accuracy via Intersection-over-Union $\mathrm { ( I o U \ge 0 . 3 ) }$ between element bounding boxes.

Together, these four dimensions evaluate whether the model faithfully reconstructs the semantic structure, content and layout of diagram objects. The formal formulation of F1 score per dimension is provided in Appendix §C.2. $F 1 _ { a v g }$ is the average across the four dimensions: $\begin{array} { r } { \frac { 1 } { 4 } \left( \mathrm { F } 1 _ { t y p e } + \mathrm { F } 1 _ { t e x t } + \mathrm { F } 1 _ { c o l o r } + \mathrm { F } 1 _ { b b o x } \right) } \end{array}$

For the editing task, we further split the code- and object-based metrics into two parts: preserve-only, computed over elements that should stay the same, and edit-only, computed over elements that the instruction asks to change.

Table 4 Evaluation settings on Diagram-MMU. TikZ search tool is built as an MCP server and prompts are abbreviated (full versions in Appendix §E).
<table><tr><td>ID</td><td>Settings</td><td>Abbreviated Prompts</td><td>Capability</td></tr><tr><td>S1</td><td>Diagram-to-code parsing</td><td>Direct coding</td><td>Foundational</td></tr><tr><td>S2</td><td>+ Objects</td><td>Use the provided perception data for coding</td><td>Agentic</td></tr><tr><td>S3</td><td>+ TikZ search tool</td><td>Use search tool for TikZ syntax and examples when necessary</td><td>Agentic</td></tr><tr><td>S4</td><td>+ Objects (model-gen.)</td><td>First perceive basic obejcts in diargam, then generate code</td><td>Agentic</td></tr><tr><td>S5</td><td>+ S2&amp;S3</td><td>Plan where to call tool in synergy with perception data</td><td>Agentic</td></tr><tr><td colspan="2">S6 Diagram-to-code editing</td><td>Direct editing</td><td>Foundational</td></tr><tr><td>S7</td><td>+ Objects</td><td>Use the provided perception data for editing</td><td>Agentic</td></tr><tr><td>S8</td><td>+ TikZ search tool</td><td>Use search tool for TikZ syntax and examples when necessary</td><td>Agentic</td></tr><tr><td>S9</td><td>+ TikZ codes (required)</td><td>First generate TikZ code of diagram, then edit code</td><td>Agentic</td></tr><tr><td></td><td>S10 + TikZ codes (optional)</td><td>Plan which diagrams require TikZ code generation for editing</td><td>Agentic</td></tr><tr><td>S11 + S7&amp;S9</td><td></td><td>Plan which diagrams require TikZ code and where to call tool for editing</td><td>Agentic</td></tr><tr><td colspan="2">S12 Diagram question answering Direct answering</td><td></td><td>Foundational</td></tr><tr><td></td><td>S13 + Objects</td><td>Use the provided perception data for answering</td><td>Agentic</td></tr><tr><td></td><td>S14 + TikZ codes (required)</td><td>First generate TikZ code of diagram, then answer question</td><td>Agentic</td></tr><tr><td></td><td>S15 + TikZ codes (optional)</td><td>Plan which diagrams require TikZ code generation for answering</td><td>Agentic</td></tr><tr><td></td><td>S16 + TikZ search tool&amp;S14</td><td>Plan which diagrams require TikZ code and where to call tool for answering Agentic</td><td></td></tr></table>

Diagram Question Answering. Following [37], we evaluate diagram question answering using accuracy. We use Qwen3-Next-80B-A3B-Instruct to extract the answer and assign binary scores (0 for incorrect, 1 for correct). The grading prompt is provided in Appendix §C.3.

## 3.5 Evaluation Scheme

In real scientific writing workflows, MLLMs working with diagrams need foundational abilities, e.g., perception, coding, domain knowledge, and reasoning, but also the ability to know when and how to act (a.k.a agentic capability): they parsing a diagram may first consult TikZ documentation for unfamiliar syntax (tool use), leverage perceptual objects in the diagram to write TikZ code (context utilization), build on generated diagram code to edit (state management), or decide which of these steps to invoke or in what order (planning). Diagram-MMU is, to our knowledge, the first benchmark to provide 16 flexible control settings evaluating both foundational and agentic capability of current MLLMs (Table 4). Specifically, the four levels of agentic ability are defined as: (1) Context utilization: whether models can leverage task-relevant information from context; (2) Tool use: knowing when to invoke tools, what to query, and how to incorporate the results; (3) State management: whether models can build on prior outputs incrementally toward the final goal; (4) Planning: deciding which ability is needed and how to combine them for task solving [38–41].

For tool use evaluation, we build our TikZ search tool as an MCP server. Naïve approaches such as web search or loading full PDF manuals introduce noise: browsers may return forum threads and advertisements, while complete manuals (e.g., the PGFPlots reference spans ∼560 pages) cause context rot. We use Mintlify<sup>1</sup> to generate an MCP server from curated TikZ documentation, enabling selective access to relevant references instead of ingesting entire documents. MCP serves as a unified tool interface, allowing any MLLM to query syntax references without model-specific implementations (details in Appendix §F).

We do not claim Diagram-MMU evaluates the full spectrum of agentic ability; as a pilot benchmark for evaluating agentic ability in vibe writing w.r.t. diagram-to-code and understanding tasks, our coverage targets only certain aspects. We discuss limitations in Appendix §H and hope to inspire future eforts to advance more thorough agentic evaluation dimensions in the vibe writing workspaces.

Results. Fig. 4 reports $\operatorname { F } 1 _ { a v g }$ for D2C-P, $\operatorname { F } 1 _ { a v g }$ (edit-only) for D2C-E, and accuracy for DQA (see §4 for detailed results and analysis.)

![](images/d3c31dcd3679f5b3db115c384b9d2a73a2f9698a207220ac6c3c172d017b87b1.jpg)

![](images/45e032bbc79ec5a8efa76ff985fa2bb1d55bc25d145b27b877277438c2517623.jpg)

![](images/2a5e6670064479750fb4d9dbaf98a1d6da738ef00a56c29cc0a368c1752129b6.jpg)  
Figure 4 Foundational vs. agentic performance across three tasks. Green/red annotations indicate gains/drops from foundational to agentic. Current models reason well over diagrams but struggle with visual perception and coding under foundational evaluation. Agency improves editing more than parsing, likely because textual instructions guide tool and context use, but nearly all models degrade on DQA.

Foundational capacity: current models show stronger capability on reasoning but weaker ability on perception and coding, revealing an interesting asymmetry: they excel in knowledge-intensive evaluations but still struggle to link visual objects to coding skills. In short, models can reason deeply over diagrams but fail to code over them–a visual perception weakness aligned with the findings of BabyVision [42]. Overall, Gemini-3.0 Pro achieves the most balanced profile.

Agentic capacity: Across the coding tasks, closed-source models show stronger agency on D2C-E than D2C-P. For example, GPT-5.2 drops -0.1 on D2C-P but gains +3.0 on D2C-E; we hypothesize that the textual editing instructions help guide models to know when to use tools and what context is useful for target editing. For DQA, unlike direct answering (input diagram → answer), in the agentic setting the model plans to invoke diagram-to-code parsing when needed (input diagram → TikZ code → answer) and decides what to query in the TikZ search tool. Except for Claude-4.6 Opus (+0.4), all models degrade on DQA, exposing weak planning ability for complex tasks, but more critically, revealing that models cannot efectively map codes to high-level reasoning.

## 4 Experiments

Evaluated Models. We evaluate 12 closed-source and open-source MLLMs on Diagram-MMU. Closed-source (6): Gemini-3.1 Pro [4], Gemini-3.0 Pro [43], Gemini-3.0 Flash [44], GPT-5.2 [6], Claude-4.6 Opus [45], and Seed-2.0 Pro [7]; Open-source (6): five general-purpose models, e.g., Qwen3.5-397B-A17B [5], Qwen3-VL-235B A22B [46], Kimi-K2.5 [47], Qwen3-VL-8B [46], and InternVL3-38B [48], and one specialist TikZero+ 10B [49] which is fine-tuned on 456K diagram–TikZ code pairs. TikZero+ 10B is not trained on editing or question answering samples, so we evaluate it only on the diagram-to-code parsing task. Generation configurations of models are provided in Table D.1.

## 4.1 Main Results

Fundamental Capacity. Table 5 presents the main results of fundamental capabilities under pass@1 evaluation (average over 6 types of diagrams; per-type results in Appendix §G.1. Pass@k evaluation across three tasks is shown in Fig. 5, where models improve with increasing $k ,$ with the largest relative gain from pass@1 to pass@2, and saturate when k exceeds 4. Table 5 reports $\mathrm { F 1 _ { a v g } }$ for D2C-P and D2C-E (per object F1 score in Fig. 6 and more analysis in Appendix § G.2).

(1) A larger performance gap among open-source models in object perception. D2C-P requires models to perceive basic objects and integrate them into code. Image-based metrics measure visual appearance similarity; all general-purpose models score highest here, and the gap between closed- and open-source models is small (e.g., closed-source models achieve 62.33–75.48 and open-source models 61.02–77.31). At coding syntax and structure evaluation (CrystalBLEU), generic models score lowest, with a similarly small gap (closed-source:

Table 5 Main results on foundational capability. Object-level metric is $\mathrm { F 1 _ { a v g } }$ (%), with preserve-only (p) and edit-only (e) split for diagram-to-code editing. Code-level metric is CrystalBLEU (%). Image-level is measured by the average of SSIM and CLIP (SC, %) and the average of FID and LPIPS (FL, %); ↓ means lower is better. All averages $\mathrm { F 1 _ { a v g } }$ CBLEU, and SC (higher is better). Diagram question answering (DQA) uses accuracy $( \mathrm { A c c . , \ \% } )$ . Bold is best in class; underline is second best.
<table><tr><td rowspan="2">Model</td><td colspan="4">Diagram-to-Code Parsing (D2C-P)</td><td colspan="7">Diagram-to-Code Editing (D2C-E)</td><td rowspan="2">DQA</td></tr><tr><td> $A l l$ </td><td> $\mathrm { F 1 _ { a v g } }$ </td><td> $\mathrm { C B L E U }$ </td><td> $\mathrm { ( S C ~ / ~ F L \downarrow ) }$ </td><td>All</td><td></td><td> $\mathrm { F 1 _ { a v g } \left( p / e \right) }$ </td><td>CBLEU</td><td> $\left( \mathrm { p / e } \right)$ </td><td> $\mathrm { ( S C ~ / ~ F L \downarrow ) }$ </td><td></td></tr><tr><td colspan="10">Close-Source Multimodal Large Language Models</td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini-3.1 Pro</td><td>48.38</td><td>49.94</td><td>32.88</td><td>(62.33</td><td>4.05)</td><td>51.22</td><td>(60.60</td><td>40.84)</td><td>(42.02</td><td>2.98)</td><td>(66.47 2.52)</td><td>86.29</td></tr><tr><td>Gemini-3.0 Pro</td><td>53.28</td><td>54.64</td><td>32.76</td><td>(72.44</td><td>4.61)</td><td>53.60</td><td>(65.12 40.24)</td><td>(44.77</td><td>2.34)</td><td></td><td>(72.38 3.12)</td><td>86.46</td></tr><tr><td>Gemini-3.0 Flash</td><td>50.97</td><td>51.19</td><td>30.67</td><td>(71.04</td><td>4.79)</td><td>49.01</td><td>(59.29 37.24)</td><td>(42.21</td><td>2.17)</td><td></td><td>(67.44 2.98)</td><td>84.07</td></tr><tr><td>GPT-5.2</td><td>51.88</td><td>51.35</td><td>28.81</td><td>(75.48</td><td>6.56)</td><td>46.15</td><td>(55.59 31.15)</td><td>(38.75</td><td>1.15)</td><td></td><td>(65.33 4.55)</td><td>81.67</td></tr><tr><td>Claude-4.6 Opus</td><td>52.41</td><td>52.23</td><td>31.27</td><td>(73.73</td><td>7.11)</td><td>51.18</td><td>(62.56 31.95)</td><td>(43.59</td><td>1.42)</td><td>(71.78</td><td>4.61)</td><td>68.75</td></tr><tr><td>Seed-2.0 Pro</td><td>50.89</td><td>47.57</td><td>32.56</td><td>(72.54</td><td>7.55)</td><td>43.73</td><td>(53.03 30.14)</td><td>(37.49</td><td>1.41)</td><td></td><td>(61.86 5.34)</td><td>75.90</td></tr><tr><td colspan="10">Open-Source Multimodal Large Language Models</td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-397B-A17B</td><td>52.72</td><td>51.38</td><td>32.92</td><td>(73.84</td><td>5.73)</td><td>50.82</td><td>(62.75 36.35)</td><td>(43.66</td><td>1.91)</td><td></td><td>(70.43 3.76)</td><td>83.42</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>55.98</td><td>55.11</td><td>37.92</td><td>(74.90</td><td>6.69)</td><td>52.25</td><td>(63.79 33.38)</td><td>(43.05</td><td>1.85)</td><td></td><td>(70.07 4.57)</td><td>63.60</td></tr><tr><td>Kimi-K2.5</td><td>56.97</td><td>57.48</td><td>36.13</td><td>(77.31</td><td>5.20)</td><td>52.13</td><td>(63.12 36.52)</td><td>(42.11</td><td>2.02)</td><td></td><td>(69.88 3.17)</td><td>79.74</td></tr><tr><td>Qwen3-VL-8B</td><td>48.26</td><td>44.66</td><td>33.31</td><td>(66.81</td><td>8.23)</td><td>19.00</td><td>(21.26 9.95)</td><td>(16.29</td><td>0.51)</td><td></td><td>(26.94 9.50)</td><td>47.12</td></tr><tr><td>InternVL3-38B</td><td>41.13</td><td>31.47</td><td>30.91</td><td>(61.02</td><td>10.14)</td><td>38.85</td><td>(44.72 22.69)</td><td>(33.13</td><td>1.29)</td><td></td><td>(57.95 7.02)</td><td>56.39</td></tr><tr><td>TikZero+ 10B</td><td>25.30</td><td>15.43</td><td>17.19</td><td>(43.29 12.67)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

28.81–32.88; open-source: 30.91–37.92). However, a larger gap emerges at object perception: closed-source $\mathrm { F 1 _ { a v g } }$ clusters within 47.57–54.64, while open-source spans 31.47–57.48, and the specialist TikZero+ trails far behind at only 15.43. This means a generated diagram may appear visually plausible at image level yet contain incorrect identification at object level.

(2) Models follow editing instructions but struggle to produce matching code. In the editing task, models reach only 0.51–2.98 CrystalBLEU on the edit partitions, yet achieve relatively higher object F1 scores, indicating they may generate the correct objects in line with the editing instructions but fail to place them at the correct positions or formats to match the ground truth code.

(3) Models failing on D2C-P also struggle with D2C-E and DU. We split diagrams into two sets conditioned on whether D2C-P produces executable TikZ code (Fig. 7). On diagrams where D2C-P succeeds, models consistently achieve higher D2C-E scores than on D2C-P failures: the object $\mathrm { F 1 _ { a v g } }$ (average of preserveonly&edit-only) gap between the two sets reaches 20–40 points across most models and code lengths. The similar pattern holds at code-level metrics. Even models can produce executable TikZ code on D2C-P failed-rendering diagrams during editing; this does not yield accurate editing. We hypothesize that such executable TikZ results from textual editing instructions, which guide code generation, e.g., leveraging the model’s stronger text-to-code ability [2, 3].

![](images/b7808a5c2f66e8f4d0397d5cfc91a45bd9caf5f685555e237268b6f9bd6b987e.jpg)

![](images/92bf1d7ddbced9e6401d9d707bb064fb571f5ae4d3da92714635c5a3e473ddc0.jpg)

![](images/4794224b5d0a5a7ddb83fdcf240e2a6eee4aced0471bee3b858b3b101a20694e.jpg)  
Gemini-3.1 Pro Gemini-3.0 Pro GPT-5.2 Claude-4.6 Opus Seed-2.0 Pro Qwen3-VL-8B  
Figure 5 Pass@k evaluation across three tasks: performance improves with $k ,$ with gains peaking from pass@1 to pass@2 and saturating beyond $k = 4$

![](images/fca6f232e892845ec76f8929f1e680100163834d08740c405bf58ec9aa3e76c3.jpg)  
Figure 6 (Updated) Breakdown of F1 scores. Each panel shows five axes (avg, type, bbox, color, text), each with its own scale. F1<sub>bbox</sub> drops sharply in Diagram-to-Code Parsing, especially for GPT-5.2, Claude-4.6 Opus, and Seed-2.0 Pro.

(4) Models struggle with fine-grained spatial grounding. All six models achieve relatively high performance on text extraction and color recognition (Fig. 6). However, spatial grounding (bbox) is consistently the weakest perception dimension. While models can recognize what elements exist, they struggle to encode where they are located. This deficit is sharpest for GPT-5.2 and Claude-4.6 Opus, where both score 62–71 on type, text, and color yet only 8.0 and 9.2 on bbox. Editing instructions in D2C-E compensate for this weakness only on preserved elements, not on newly edited ones: Claude-4.6 Opus’s $\mathrm { F 1 } _ { \mathrm { b b o x } }$ rises from 9.2 to 24.3 on preserve, and GPT-5.2 from 8.0 to 20.8; however, edit-only $\operatorname { F 1 } _ { \mathrm { b b o x } }$ drops to 3.0 and 2.3, respectively, since placing newly introduced elements is harder than retaining existing ones. Qwen3-VL-8B, however, falls behind all models across every perception dimension.

(5) Models struggle most with 3D shapes across all three tasks. 3D shapes pose the greatest challenge for both closed- and open-source models across all three tasks, likely due to the scarcity of TikZ-3D training samples in pretraining corpora. For D2C-P and D2C-E, charts are the second most challenging domain after 3D, as they contain a higher density of small-scale elements (e.g., tick labels, data markers). Model-specific DQA weaknesses diverge beyond 3D: Seed-2.0 Pro drops notably on charts; Claude-4.6 Opus underperforms on graph structures; and Gemini-3.0 Pro shows relatively lower accuracy on circuits (see Table G.3 in Appendix §G.3).

Agentic Capacity. We select a mini split of 300 diagrams (50 per domain) with 1,500 QA instances (300/600/600 for D2C-P/D2C-E/DQA), evaluated on 6 representative models: Seed-2.0 Pro, Claude-4.6 Opus, GPT-5.2, Gemini-3.0 Pro, Gemini-3.1 Pro, and Qwen3-VL-8B.

(1) Context utilization. We provide diagram objects as context across all three tasks (Tables 6 and 7). On D2C-E, objects improve all models consistently: $\mathrm { F 1 _ { a v g } }$ gains 4.1–10.6 on preserve and 3.3–7.1 on edit; thus objects may help models locate which elements to keep and which to modify. On D2C-P, results are mixed: four models gain, while Gemini-3.0 Pro (-2.0) and Qwen3-VL-8B (-6.1) degrade. At code level, only Claude-4.6 Opus (+0.3) and Gemini-3.1 Pro (+1.4) gain, showing objects aid element perception more than code synthesis. On DQA, most models benefit, while GPT-5.2 drops sharply (-7.3), indicating it cannot map low-level objects to semantic reasoning.

Takeaway: Most models can use objects to ground local edits but struggle to integrate them into syntax or domian knowledege reasoning.

![](images/b63c62a8dfb2f9e6d56d65a0719a4682802c9c0629c47d12302d95b7a1486126.jpg)

![](images/321b2ed901a0d2acf130ccceaeeab299701be0b628d530a7f299dcf471d37e1b.jpg)

![](images/288855e2d75d2783aee7efeba72fbe5b7164eb6a169b51af682d92b8829aa890.jpg)  
Gemini-3.1 Pro Gemini-3.0 Pro GPT-5.2 Claude-4.6 Opus Seed-2.0 Pro Qwen3-VL-8B  
Figure 7 (Updated) Model performance on D2C-E/DQA across two diagram sets split by D2C-P output: successful rendering (solid lines) vs. failed rendering (dashed lines). D2C-E is evaluated by object $\mathrm { F 1 _ { a v g } }$ (left) and CrystalBLEU (middle), and DQA by accuracy (right). The x-axis denotes ground-truth code length. Models that fail D2C-P also score lower on D2C-E and DQA.

Table 6 Main results on agentic evaluation settings across D2C-P (S1–S5) and DQA (S12–S16). S1/S12 uses direct coding/answering; Context utilization: S2/S13 provide diagram objects as context; Tool use: S3 provides TikZ search tool; State management: S4 perceives objects first, then builds upon them to code; S14 requires answering building upon the coding states; Planning: S5 coordinates objects and tool; S15 optionally generates code before answering; S16 adds tool access to S15 (Table 4). ∆ (↑ gain, ↓ drop) over direct coding/answering S1/S12.
<table><tr><td rowspan="2">Model</td><td colspan="2">S1</td><td colspan="2">S2</td><td colspan="2">S3</td><td colspan="2">S4</td><td colspan="2">S5</td><td colspan="2">|S12 S13</td><td colspan="2">S14</td><td colspan="2">S15 S16</td></tr><tr><td> $\mathrm { F 1 _ { a v g } }$ </td><td>CBLEU</td><td> $\mathrm { F 1 _ { a v g } }$ </td><td>CBLEU</td><td> $\mathrm { F 1 _ { a v g } }$ </td><td>CBLEU</td><td> $\mathrm { F 1 _ { a v g } }$ </td><td>CBLEU</td><td> $\mathrm { F 1 _ { a v g } }$ </td><td></td><td>CBLEU|</td><td> $\operatorname { A c c } .$ </td><td>| Acc. |</td><td>Acc.</td><td>|Acc.</td><td>Acc.</td></tr><tr><td>Seed-2.0 Pro</td><td>44.4</td><td>31.6</td><td>↑0.2</td><td>↓0.2</td><td>↑1.2</td><td>↑1.4</td><td>↓1.2</td><td>↓1.2</td><td>↑2.6</td><td>↑1.5</td><td></td><td>86.0</td><td>↑1.0</td><td>↑0.8</td><td>↓1.8</td><td>↓8.8</td></tr><tr><td>Claude-4.6 Opus</td><td>48.5</td><td>30.6</td><td>↑1.2</td><td>↑0.3</td><td>↑2.2</td><td>↑1.9</td><td>↓0.2</td><td>↓0.9</td><td>↑5.1</td><td></td><td>↑2.8</td><td>80.7</td><td>0.0</td><td>↑3.8</td><td>↓2.0</td><td>↓0.3</td></tr><tr><td>GPT-5.2</td><td>48.9</td><td>27.9</td><td>↑2.9</td><td>↓2.9</td><td>↓0.3</td><td>↓0.5</td><td>↓1.5</td><td>↓0.8</td><td>↓1.6</td><td>↓4.1</td><td></td><td>87.7</td><td>↓7.3</td><td>↓1.3</td><td>↓4.3</td><td>↓5.7</td></tr><tr><td>Gemini-3.0 Pro</td><td>50.9</td><td>30.8</td><td>↓2.0</td><td>↓3.2</td><td>↓1.0</td><td>↓0.6</td><td>↓1.3</td><td>↓1.2</td><td>↑0.1</td><td>↓0.5</td><td></td><td>88.3</td><td>↑1.2</td><td>↓2.5</td><td>↓2.3</td><td>↓0.3</td></tr><tr><td>Gemini-3.1 Pro</td><td>47.1</td><td>31.6</td><td>↑1.9</td><td>↑1.4</td><td>↓4.3</td><td>↓3.0</td><td>↑0.3</td><td>↓0.8</td><td>↓5.0</td><td>↓4.5</td><td></td><td>90.2</td><td>↑0.5</td><td>↓3.2</td><td>↓2.7</td><td>0.0</td></tr><tr><td>Qwen3-VL-8B</td><td>40.7</td><td>34.1</td><td>↓6.1</td><td>↓5.6</td><td>↓0.1</td><td>↓1.2</td><td>↑1.4</td><td>↓0.2</td><td>↓5.6</td><td>↓6.4</td><td></td><td>50.7</td><td>↑3.0</td><td>↓1.3</td><td>↓2.3</td><td>↓3.3</td></tr></table>

(2) Tool use. Most models exhibit weak tool-calling ability. The sharpest failure is Gemini-3.1 Pro (-4.3 F1<sub>avg</sub> on D2C-P). Manual inspection of trajectory reveals that model iteratively queries without a clear stopping criterion, causing context rot until terminated by the token-length limit (tool-call counts and failure rates in Appendix §G.4); Qwen3-VL-8B trajectory issues poorly targeted queries, lacking ability of what to query in the search tool. Claude-4.6 Opus performs best, with consistent gains on both DC $( + 2 . 2 \mathrm { F } 1 _ { a v g } )$ and DE (+4.4/+2.2).

Takeaway: Only Claude-4.6 Opus consistently benefits from tool access; Gemini-3.1 Pro sufers from excessive retrieval; Qwen3-VL-8B fails at querying.

(3) State management. We require models to first generate the diagram’s TikZ code, then build upon it for editing (D2C-E) or answering (DQA). On D2C-E, most models degrade on preserve $\mathrm { F 1 _ { a v g } }$ , indicating the limitation of managing intermediate code states for precise editing. Claude-4.6 Opus $( + 2 . 5 / + 2 . 6 )$ and GPT-5.2 $\left( + 2 . 5 / + 3 . 5 \right)$ are exceptions, successfully building on their generated code. On DQA, Claude-4.6 Opus gains substantially (+3.8) and Seed-2.0 Pro gains marginally (+0.8), while all other models degrade (-1.3 to -3.2).

Takeaway: Most models fail to manage intermediate code states across steps; only Claude-4.6 Opus maintains coherence from coding to editing&reasoning.

(4) Planning. We test whether models can coordinate multiple agentic abilities: combining objects with tool use for D2C-P, optionally generating code before editing or answering, and coordinating tool use with optional code generation (D2C-E and DU). On D2C-P, Claude-4.6 Opus (+5.1/+2.8) and Seed-2.0 Pro $\left( + 2 . 6 / + 1 . 5 \right)$ coordinate both resources successfully, while (-1.6/-4.1), Gemini-3.1 Pro (-5.0/-4.5), and Qwen3- VL-8B (-5.6/-6.4) degrade. On D2C-E, most models show marginal or negative changes under optional code generation; adding tool access recovers Claude-4.6 Opus $\left( + 4 . 1 / + 3 . 0 \right)$ and GPT-5.2 (+2.6/+1.9), but degrades Gemini-3.1 Pro (-4.4/-0.9). On DQA, all six models degrade under optional code generation, and adding tool access amplifies degradation for Seed-2.0 Pro (-8.8) and GPT-5.2 (-5.7).

Table 7 Agentic evaluation results across D2C-E (S6–S11) with preserve (p) / edit (e) split. S6 uses direct editing; Context utilization: S7 provides diagram objects as context; Tool use: S8 provides TikZ search tool; State management: S9 requires editing building upon the coding states; Planning: S10 optionally generates code for editing; S11 adds tool access to S10 (Table 4). ∆ (↑ gain, ↓ drop) over direct editing.
<table><tr><td rowspan="2">Model</td><td colspan="2">S6</td><td colspan="2">S7</td><td colspan="2">S8</td><td colspan="2">S9</td><td colspan="2">S10</td><td colspan="2">S11</td></tr><tr><td colspan="2"> $\left| \mathrm { F 1 } _ { \mathrm { a v g } } ( \mathrm { p / e } ) \right|$ </td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2">CBLEU(p/e)|F1avg(p/e) CBLEU(p/e) |F1avg(p/e) CBLEU(p/e) |F1avg(p/e) CBLEU(p/e) |F1avg(p/e) CBLEU(p/e)|F1avg(p/e) CBLEU(p/e)</td><td colspan="2"></td></tr><tr><td>Seed-2.0 Pro</td><td>50.6/28.0</td><td>34.3/1.5</td><td>↑9.3/↑7.1</td><td>↑6.6/↑0.5</td><td>↑0.7/↑0.2</td><td>↑0.7/↑0.1</td><td>↓2.3/↓0.4</td><td>↓1.3/↑0.1</td><td>0.0/↓0.4</td><td>↑0.1/↓0.3</td><td>↑0.6/↑0.6</td><td>↑0.1/0.0</td></tr><tr><td>Claude-4.6 Opus</td><td>55.9/28.3</td><td>39.3/1.3</td><td>↑8.9/↑6.0</td><td>↑6.0/0.0</td><td>↑4.4/↑2.2</td><td>↑4.7/↑0.5</td><td>↑2.5/↑2.6</td><td>↑2.5/0.0</td><td>↓1.5/↓0.2</td><td>↓0.4/↑0.1</td><td>↑4.1/↑3.0</td><td>↑5.6/↑0.7</td></tr><tr><td>GPT-5.2</td><td>54.9/29.3</td><td>38.6/1.2</td><td>↑10.6/↑5.9</td><td>↑7.2/↑0.1</td><td>↑1.3/↑2.6</td><td>↑2.4/0.0</td><td>↑2.5/↑3.5</td><td>↑2.7/↑0.1</td><td>↑0.9/↑1.1</td><td>↑1.6/↑0.3</td><td>↑2.6/↑1.9</td><td>↑2.9/↑0.4</td></tr><tr><td>Gemini-3.0 Pro</td><td>58.8/36.1</td><td>40.9/1.9</td><td>↑4.1/↑3.3</td><td>↑2.4/↑0.4</td><td>↑2.2/↓0.1</td><td>↓0.1/↑0.5</td><td>↓1.1/↓0.6</td><td>↓2.0/↑0.2</td><td>↓0.4/↓0.5</td><td>↓1.7/↑0.2</td><td>↑2.8/↑0.8</td><td>↑0.7/↑0.5</td></tr><tr><td>Gemini-3.1 Pro</td><td>57.6/38.1</td><td>38.0/2.5</td><td>↑6.6/↑4.8</td><td>↑4.6/↑0.6</td><td>↓8.5/↓3.5</td><td>↓5.3/↑0.8</td><td>↓5.2/↓3.7</td><td>↓2.9/↑0.2</td><td>↑0.6/↑1.7</td><td>↑0.5/↑0.7</td><td>↓4.4/↓0.9</td><td>↓2.9/↑1.4</td></tr><tr><td>Qwen3-VL-8B</td><td>47.0/22.2</td><td>34.2/0.8</td><td>↑8.4/↑3.3</td><td>↑4.3/↑0.2</td><td>↓2.0/↓0.2</td><td>↓2.2/↑0.2</td><td>↓1.0/↑2.5</td><td>↓1.4/↑0.4</td><td>↓3.3/↓1.0</td><td>↓3.0/0.0</td><td>0.0/↑1.7</td><td>↓0.9/↑0.3</td></tr></table>

Takeaway: Planning is the weakest agentic capability; no model reliably composes multiple abilities as task complexity grows from D2C-P to DQA.

## 5 Conclusion

Scientific diagrams are an important visual medium for expressing abstract ideas, and their TikZ code representation can be natively compiled inline within LAT<sub>E</sub>X during scientific paper writing. Vibe writing platforms provide MLLM-assisted paper writing, and one important feature is converting scientific diagrams directly into TikZ code. We present Diagram-MMU, a TikZ-based benchmark with 16 evaluation settings across diagram-to-code parsing, editing, and understanding alongside agentic settings per task. We also build a TikZ search tool as an MCP server that any model can selectively query TikZ syntactic references without any model-specific implementations. Comprehensive evaluation of 12 MLLMs reveals their weaknesses in coding diagrams into TikZ, and stronger agentic abilities, e.g., context utilization and tool use, could help. We hope our analysis can inspire further research to build more thorough evaluation settings and develop methods to improve MLLM-assisted vibe writing workspaces.

## Acknowledgements

We thank all 13 annotators for their contributions to data quality assurance. Collectively, they cross-validated over 2,000 scientific diagrams across six domains. Three annotators are co-authors. The ten non-author contributors include one research scientist from NetMind.ai, eight PhD students from Nanjing University of Science and Technology (NJUST), and one PhD student from Westlake University. Kindly refer to § I for details of their contributions.

## References

[1] Minimax. Minimax m2.5: Built for real-world productivity. https://www.minimax.io/news/minimax-m25, 2026.

[2] Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

[3] Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chengxing Xie, Cunxiang Wang, et al. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

[4] Google DeepMind. Gemini 3.1 pro model card. https://storage.googleapis.com/deepmind-media/ Model-Cards/Gemini-3-1-Pro-Model-Card.pdf, 2026.

[5] Qwen Team. Qwen3.5: Towards native multimodal agents. https://qwen.ai/blog?id=qwen3.5, 2026.

[6] OpenAI. Gpt-5.2. https://developers.openai.com/api/docs/models/gpt-5.2, 2025.

[7] Bytedance Seed. Seed2.0 model card: Towards intelligence frontier for real-world complexity. https://lf3-static.bytednsdoc.com/obj/eden-cn/lapzild-tss/ljhwZthlaukjlkulzlp/seed2/0214/ Seed2.0%20Model%20Card.pdf, 2026.

[8] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9556–9567, 2024.

[9] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15134–15186, 2025.

[10] Yanpeng Sun, Jing Hao, Ke Zhu, Jiang-Jiang Liu, Yuxiang Zhao, Xiaofan Li, Gang Zhang, Zechao Li, and Jingdong Wang. Descriptive caption enhancement with visual specialists for multimodal perception. arXiv preprint arXiv:2412.14233, 2024.

[11] Weihao Bo, Shan Zhang, Yanpeng Sun, Jingjing Wu, Qunyi Xie, Xiao Tan, Kunbin Chen, Wei He, Xiaofan Li, Na Zhao, et al. Agentic learner with grow-and-refine multimodal semantic memory. arXiv preprint arXiv:2511.21678, 2025.

[12] OpenAI. Prism. https://openai.com/prism/, 2026.

[13] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems, 37:113569–113697, 2024.

[14] Cheng Yang, Chufan Shi, Yaxin Liu, Bo Shui, Junjie Wang, Mohan Jing, Linran XU, Xinyu Zhu, Siheng Li, Yuxiang Zhang, et al. Chartmimic: Evaluating lmm’s cross-modal reasoning capability via chart-to-code generation. In The Thirteenth International Conference on Learning Representations, 2025.

[15] Yanpeng Sun, Shan Zhang, Wei Tang, Aotian Chen, Piotr Koniusz, Kai Zou, Yuan Xue, and Anton van den Hengel. Math blind: Failures in diagram understanding undermine reasoning in mllms. arXiv preprint arXiv:2503.20745, 2025.

[16] Lingjie Jiang, Shaohan Huang, Xun Wu, Yixia Li, Dongdong Zhang, and Furu Wei. Viscodex: Unified multimodal code generation via merging vision and coding models. arXiv preprint arXiv:2508.09945, 2025.

[17] Chengyue Wu, Zhixuan Liang, Yixiao Ge, Qiushan Guo, Zeyu Lu, Jiahao Wang, Ying Shan, and Ping Luo. Plot2code: A comprehensive benchmark for evaluating multi-modal large language models in code generation from scientific plots. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 3006–3028, 2025.

[18] Juan A Rodriguez, Abhay Puri, Shubham Agarwal, Issam H Laradji, Pau Rodriguez, Sai Rajeswar, David Vazquez, Christopher Pal, and Marco Pedersoli. Starvector: Generating scalable vector graphics code from images and text. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16175–16186, 2025.

[19] Jingxuan Wei, Cheng Tan, Qi Chen, Gaowei Wu, Siyuan Li, Zhangyang Gao, Linzhuang Sun, Bihui Yu, and Ruifeng Guo. From words to structured visuals: A benchmark and framework for text-to-diagram generation and editing. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13315–13325, 2025.

[20] Jonas Belouadi, Anne Lauscher, and Stefen Eger. Automatikz: Text-guided synthesis of scientific vector graphics with tikz. In The Twelfth International Conference on Learning Representations.

[21] Jonas Belouadi, Simone Ponzetto, and Stefen Eger. Detikzify: Synthesizing graphics programs for scientific figures and sketches with tikz. Advances in Neural Information Processing Systems, 37:85074–85108, 2024.

[22] Josselin S Roberts, Tony Lee, Chi H Wong, Michihiro Yasunaga, Yifan Mai, and Percy Liang. Image2struct: Benchmarking structure extraction for vision-language models. Advances in Neural Information Processing Systems, 37:115058–115097, 2024.

[23] Shuo Li, Jiajun Sun, Zhekai Wang, Xiaoran Fan, Hui Li, Dingwen Yang, Zhiheng Xi, Yijun Wang, Zifei Shan, Tao Gui, et al. ChartE<sup>3</sup>: A comprehensive benchmark for end-to-end chart editing. arXiv preprint arXiv:2601.21694, 2026.

[24] Donglu Yang, Liang Zhang, Zihao Yue, Liangyu Chen, Yichen Xu, Wenxuan Wang, and Qin Jin. Chartm3: Benchmarking chart editing with multimodal instructions. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 5001–5009, 2025.

[25] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pages 2263–2279, 2022.

[26] Zeju Qiu, Weiyang Liu, Haiwen Feng, Zhen Liu, Tim Z Xiao, Katherine M Collins, Joshua B Tenenbaum, Adrian Weller, Michael J Black, and Bernhard Schölkopf. Can large language models understand symbolic graphics programs? In The Thirteenth International Conference on Learning Representations, 2025.

[27] Anthropic. Model context protocol. https://modelcontextprotocol.io, 2024.

[28] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024.

[29] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24108–24118, 2025.

[30] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations

[31] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024.

[32] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

[33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[34] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

[35] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

[36] Aryaz Eghbali and Michael Pradel. Crystalbleu: precisely and eficiently measuring the similarity of code. In Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering, pages 1–12, 2022.

[37] Runjie Zhou, Youbo Shao, Haoyu Lu, Bowei Xing, Tongtong Bai, Yujie Chen, Jie Zhao, Lin Sui, Haotian Yao, Zijia Zhao, et al. Worldvqa: Measuring atomic world knowledge in multimodal large language models. arXiv preprint arXiv:2602.02537, 2026.

[38] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.

[39] Rajkumar Buyya et al. Agentic artificial intelligence (ai): Architectures, taxonomies, and evaluation of large language model agents. arXiv preprint arXiv:2601.12560, 2026.

[40] Thomas Hartung. Ai, agentic models and lab automation for scientific discovery—the beginning of scaince. Frontiers in Artificial Intelligence, 8:1649155, 2025.

[41] Shixiang Tang, Yizhou Wang, Lu Chen, Yuan Wang, Sida Peng, Dan Xu, and Wanli Ouyang. Human-centric foundation models: Perception, generation and agentic modeling. arXiv preprint arXiv:2502.08556, 2025.

[42] Liang Chen, Weichu Xie, Yiyan Liang, Hongfeng He, Hans Zhao, Zhibo Yang, Zhiqi Huang, Haoning Wu, Haoyu Lu, Yiping Bao, et al. Babyvision: Visual reasoning beyond language. arXiv preprint arXiv:2601.06521, 2026.

[43] Google DeepMind. Gemini 3 pro model card. https://storage.googleapis.com/deepmind-media/Model-Cards/ Gemini-3-Pro-Model-Card.pdf, 2025.

[44] Google DeepMind. Gemini 3 flash model card. https://storage.googleapis.com/deepmind-media/ Model-Cards/Gemini-3-Flash-Model-Card.pdf, 2025.

[45] Anthropic. Claude opus 4.6. https://www.anthropic.com/claude/opus, 2026.

[46] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[47] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. Kimi k2.5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

[48] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

[49] Jonas Belouadi, Eddy Ilg, Margret Keuper, Hideki Tanaka, Masao Utiyama, Raj Dabre, Stefen Eger, and Simone Ponzetto. Tikzero: Zero-shot text-guided graphics program synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17793–17806, 2025.

[50] Haruto Yoshida, Keito Kudo, Yoichi Aoki, Ryota Tanaka, Itsumi Saito, Keisuke Sakaguchi, and Kentaro Inui. Nodes are early, edges are late: Probing diagram representations in large vision-language models. arXiv preprint arXiv:2603.02865, 2026.

[51] Xue Li, Yiyou Sun, Wei Cheng, Yinglun Zhu, and Haifeng Chen. Chain-of-region: Visual language models need details for diagram analysis. In The Thirteenth International Conference on Learning Representations, 2025.

[52] Mojdeh Rahmanian, Ashkan Sami, and Yanchao Yu. Challenges and feasibility of multimodal llms in er diagram evaluation. Cogent Education, 12(1):2590901, 2025.

[53] Sakthivel Thangaraj, Neelesh Kumar Shukla, and Viji Krishnamurthy. Ontology-driven multimodal framework for automated interpretation and description of architecture diagrams. In 2025 IEEE International Conference on Big Data (BigData), pages 2493–2502. IEEE, 2025.

[54] Shue Shiinoki, Ryo Koshihara, Hayato Motegi, and Masumi Morishige. Overcoming vision language model challenges in diagram understanding: A proof-of-concept with xml-driven large language models solutions. arXiv preprint arXiv:2502.04389, 2025.

[55] Bowen Yu and Cláudio T. Silva. Flowsense: A natural language interface for visual data exploration within a dataflow system. IEEE Transactions on Visualization and Computer Graphics, 26(1):1–11, 2020.

[56] Qian Wang, Aleksandar Cvejić, Abdelrahman Eldesokey, and Peter Wonka. Editclip: Representation learning for image editing. pages 15960–15970, 2025.

[57] Cheng Tan, Qi Chen, Jingxuan Wei, Gaowei Wu, Zhangyang Gao, Siyuan Li, Bihui Yu, Ruifeng Guo, and Stan Z Li. Sketchagent: Generating structured diagrams from hand-drawn sketches. 2025.

[58] ZENG Xingchen, Zhewei Su, Hengming Zhang, Juyong Jiang, Jiazhi Xia, and Wei Zeng. Davinci: Reinforcing visual-structural syntax in mllms for generalized scientific diagram parsing. In The Fourteenth International Conference on Learning Representations, 2026.

[59] Qingyang Mao, Qi Cai, Yehao Li, Yingwei Pan, Mingyue Cheng, Ting Yao, Qi Liu, and Tao Mei. Visual autoregressive modeling for instruction-guided image editing. arXiv preprint arXiv:2508.15772, 2025.

[60] Christian Meske, Tobias Hermanns, Esther Von der Weiden, Kai-Uwe Loser, and Thorsten Berger. Vibe coding as a reconfiguration of intent mediation in software development: Definition, implications, and research agenda. IEEE Access, 13:213242–213259, 2025.

[61] Christian Greisinger and Stefen Eger. Tikzilla: Scaling text-to-tikz with high-quality data and reinforcement learning. arXiv preprint arXiv:2603.03072, 2026.

[62] Ke Wang, Junting Pan, Linda Wei, Aojun Zhou, Weikang Shi, Zimu Lu, Han Xiao, Yunqiao Yang, Houxing Ren, Mingjie Zhan, et al. Mathcoder-vl: Bridging vision and code for enhanced multimodal mathematical reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, pages 2505–2534, 2025.

[63] Sher Badshah, Moamen Moustafa, and Hassan Sajjad. Clev: Llm-based evaluation through lightweight eficient voting for free-form question-answering. In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pages 1513–1531, 2025.

## Diagram-MMU: A Multi-Modal Benchmark for Scientific Diagrams (Appendix)

## Contents

A Related Works . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20   
A.1 Diagram Parsing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20   
A.2 Diagram Editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20   
A.3 Vibe Writing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20   
B Benchmark Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20   
B.1 Data Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20   
B.2 Task Generation Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22   
B.3 D2C-P Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23   
B.4 D2C-E Templates . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26   
B.5 D2C-E Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27   
B.6 DQA Templates . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31   
B.7 DQA Descriptive Question Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36   
B.8 DQA Reasoning Question Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38   
C Evaluation Methodology . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40   
C.1 Semantic Object Model (SOM) Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40   
C.2 Evaluation Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41   
C.3 DQA LLM-as-a-Judge Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43   
C.4 LLM Judge Human Agreement Validation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44   
D Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45   
D.1 Model Configurations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45   
D.2 Inference and Compilation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45   
E Evaluation Prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46   
E.1 Foundational Evaluation Prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46   
F MCP-Based TikZ Documentation Server . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 49   
F.1 Server Architecture and Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 49   
F.2 Indexed Documentation List . . . . 50

## vilocean

G Additional Experimental Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51   
G.1 Foundational Results Per Diagram Type . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51   
G.2 F1 Scores Per Object . . . . . . . .   
G.3 DQA Analysis Per Diagram Type . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 52   
G.4 Agentic Tool-Call Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 52   
H Limitations and Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 54   
I Acknowledgement . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

## A Related Works

## A.1 Diagram Parsing

Diagram parsing extracts the topology of a diagram, e.g., the basic elements and their relationships. This requires symbolic perception instead of pixel-level understanding. A geometric primitive carries meaning as a symbol: its semantics come from its type, path and spatial relationships, not from its pixel values. For example, a circle remains a circle regardless of its color, size or line thickness. This contrasts with natural images, where each pixel contributes semantic content through color, texture, and intensity. This distinction explains why MLLMs that perform well on natural image tasks struggle with diagrams [15]. A common surrogate is Optical Character Recognition (OCR), where models extract text annotations to interpret diagrams. However, OCR captures only textual content while ignoring geometric topology, thus failing at structural relationship extraction [50, 51]. Consequently, error rates remain high for domain-specific diagrams like scientific charts and Entity-Relationship (ER) schemas [52]. To address this, recent work integrates visual parsing with domain ontologies [53] or uses Region Decomposition [54]. These methods capture cross-node dependencies, transforming pixel-level features into high-level semantics. An alternative line of work represents diagrams as code, called diagram-to-code parsing, which is the focus of this paper. We review these methods in §2&Tabel 1.

## A.2 Diagram Editing

Diagram editing is shifting from manual graphical manipulation to intent-driven editing using natural language [19, 55]. Unlike general image editing via difusion models, editing technical diagrams requires strict global logical coherence. Models must modify local attributes without causing structural damage [56]. Recent frameworks solve this by parsing instructions and modifying vector graphics or code directly [57]. For example, DiagramAgent uses a multi-agent system. It decomposes user instructions into explicit plans, allowing specialized agents to collaboratively edit and verify LAT<sub>E</sub>X/TikZ or DOT representations [58]. Other methods combine visual representation learning with autoregressive frameworks, such as VAREdit [59]. These approaches bridge the gap between language intents and vector graphics, safely handling complex topological changes.

## A.3 Vibe Writing

With vibe coding [60], researchers provide a diagram and describe the desired changes in natural language, and an MLLM generates the compilable code; a workflow that fundamentally depends on strong diagram-tocode generation. In the vide writing environment, human creators no longer need to write tedious macros or calculate coordinates. By providing simple logical intents or vibes, users guide multimodal agents to autonomously handle structural inference, code synthesis, and visual rendering. This radically reshapes how researchers create academic literature and diagrams.

Generating highly precise TikZ/LAT<sub>E</sub>X code is a major research focus [20]. However, high-quality diagram-code pairs are scarce. To address this, models like TikZero use zero-shot architectures to decouple visual and textual features [49]. Similarly, DeTikZify optimizes generated TikZ code using Monte Carlo Tree Search (MCTS) and compiler feedback [21]. Furthermore, TikZilla improves visual fidelity using Reinforcement Learning (RL) and inverse graphics rewards [61], while MathCoder-VL synthesizes massive aligned datasets using a “model-in-the-loop” strategy [62].

## B Benchmark Details

## B.1 Data Statistics

Diagram-MMU covers six scientific diagram types, each associated with one or more dedicated TikZ packages. The full dataset contains 3,744 diagrams paired with 18,305 evaluation instances (1 D2C-P + 2 D2C-E + 2 DQA per diagram), with a balanced mini split of 300 diagrams (50 per domain). Table B.1 summarizes the per-domain source distribution and task instance counts. We describe each type of diagram, as follows:

## (1) Diagram types:

Table B.1 Per-domain statistics of the Diagram-MMU. “Oficial” denotes diagrams sourced from TikZ package documentation; “Community” denotes diagrams from community resources. Each diagram is paired with up to 5 evaluation questions (1 D2C-P + 2 D2C-E + 2 DQA); a small number of instances were removed during human cross-validation, so per-task totals are slightly below the nominal 1×/2×/2× diagram count.
<table><tr><td></td><td></td><td></td><td colspan="2">Data Source</td><td colspan="3">Task Instances</td><td colspan="3">Code Stats</td></tr><tr><td>Domain</td><td>#Diag.</td><td>Primary TikZ Pkg.</td><td>Official</td><td>Community</td><td>D2C-P</td><td>D2C-E</td><td>DQA</td><td>Total</td><td>Avg. Lines</td><td>Avg. Chars</td></tr><tr><td>Charts</td><td>960</td><td>pgfplots</td><td>579</td><td>381</td><td>959</td><td>1,912</td><td>1,834</td><td>4,705</td><td>34.6</td><td>1,098</td></tr><tr><td>Planar Geom.</td><td>601</td><td>tikz/tkz-euclide</td><td>253</td><td>348</td><td>598</td><td>1,162</td><td>1,118</td><td>2,878</td><td>26.9</td><td>1,074</td></tr><tr><td>3D Shapes</td><td>237</td><td>pgfplots</td><td>146</td><td>91</td><td>236</td><td>471</td><td>455</td><td>1,162</td><td>29.6</td><td>988</td></tr><tr><td>Graph Struct.</td><td>1,356</td><td>tikz</td><td>299</td><td>1,057</td><td>1,356</td><td>2,699</td><td>2,679</td><td>6,734</td><td>34.9</td><td>1,452</td></tr><tr><td>Chemistry</td><td>187</td><td>chemfig</td><td>185</td><td>2</td><td>187</td><td>373</td><td>366</td><td>926</td><td>18.5</td><td>412</td></tr><tr><td>Circuit</td><td>403</td><td>circuitikz</td><td>387</td><td>16</td><td>403</td><td>803</td><td>694</td><td>1,900</td><td>21.2</td><td>660</td></tr><tr><td>Total</td><td>3,744</td><td></td><td>1,849</td><td>1,895</td><td>3,739</td><td>7,420</td><td>7,146</td><td>18,305</td><td>30.9</td><td>1,134</td></tr></table>

Table B.2 D2C-E editing dimension breakdown for the full dataset and mini split. 3D Shapes and Chemistry have no layout-dimension edits (marked “–”).
<table><tr><td colspan="7">Full Dataset</td><td colspan="4">Mini Split</td></tr><tr><td>Domain</td><td>Text</td><td>Color</td><td>Scope</td><td>Layout</td><td>Total</td><td>Text</td><td>Color</td><td>Scope</td><td>Layout</td><td>Total</td></tr><tr><td>Charts</td><td>438</td><td>194</td><td>365</td><td>915</td><td>1,912</td><td>21</td><td>10</td><td>19</td><td>50</td><td>100</td></tr><tr><td>Planar Geom.</td><td>172</td><td>205</td><td>221</td><td>564</td><td>1,162</td><td>20</td><td>17</td><td>14</td><td>49</td><td>100</td></tr><tr><td>3D Shapes</td><td>81</td><td>123</td><td>267</td><td></td><td>471</td><td>14</td><td>23</td><td>63</td><td>一</td><td>100</td></tr><tr><td>Graph Struct.</td><td>315</td><td>319</td><td>735</td><td>1,330</td><td>2,699</td><td>10</td><td>12</td><td>28</td><td>50</td><td>100</td></tr><tr><td>Chemistry</td><td>57</td><td>139</td><td>177</td><td></td><td>373</td><td>9</td><td>41</td><td>50</td><td>一</td><td>100</td></tr><tr><td>Circuit</td><td>122</td><td>71</td><td>215</td><td>395</td><td>803</td><td>18</td><td>7</td><td>26</td><td>49</td><td>100</td></tr><tr><td>Total</td><td>1,185</td><td>1,051</td><td>1,980</td><td>3,204</td><td>7,420</td><td>92</td><td>110</td><td>200</td><td>198</td><td>600</td></tr></table>

• Charts (960 diagrams). Statistical and data visualizations rendered primarily with pgfplots, including line plots, bar charts, scatter plots, pie charts, and histograms. Domain knowledge includes statistics, trend analysis, and extremum identification.

• Planar Geometry (601 diagrams). Two-dimensional geometric constructions drawn with tikz and tkz-euclide, including triangles, circles, polygons, and angle/distance annotations. Domain knowledge covers Euclidean notation, area/perimeter formulas, and geometric theorems.

• 3D Shapes (237 diagrams). Three-dimensional solid geometry visualizations rendered with pgfplots and tikz, including prisms, pyramids, spheres, and cross-section illustrations. Domain knowledge includes solid geometry, surface area, and volume computation.

• Graph Structures (1,356 diagrams). Combinatorial and network-theoretic diagrams drawn with tikz and tikz-network, including directed/undirected graphs, trees, automata, and flow networks. Domain knowledge includes degree, connectivity, shortest paths, and tree properties.

• Chemistry (187 diagrams). Molecular and chemical structure diagrams rendered with chemfig, including organic molecules, functional groups, and reaction schemes. Domain knowledge covers bond types, atom labels, and functional group identification.

• Circuit Diagrams (403 diagrams). Electrical and electronic circuit schematics drawn with circuitikz, including resistive networks, RC/RLC circuits, and logic gates. Domain knowledge includes circuit topology, Ohm’s law, and series/parallel analysis.

(2) D2C-E editing dimensions. Each D2C-E sample is annotated with one of four editing dimensions: text (label/annotation modification), color (fill/stroke color change), scope (local element addition, deletion, or transformation), and layout (global structural change such as chart type conversion or circuit topology modification). We exclude layout editing from 3D shapes and chemistry. Table B.2 shows the per-domain breakdown.

(3) DQA question types. DQA comprises two question types: descriptive and reasoning. Reasoning questions are further split into standard and what-if types (definitions in §3.3). Table B.3 shows the per-domain breakdown.

(4) DQA answer formats. Each DQA question specifies one of three output instruction types for answering:

Table B.3 DQA question type breakdown for the full dataset and mini split. Standard questions cover both descriptive and reasoning types; what-if questions are reasoning-only and require hypothetical inference conditioned on element modifications.
<table><tr><td rowspan="2">Domain</td><td colspan="4">Full Dataset</td><td colspan="4">Mini Split</td></tr><tr><td>Standard</td><td>What-if</td><td>Total</td><td>% What-if</td><td>Standard</td><td>What-if</td><td>Total</td><td>% What-if</td></tr><tr><td>Charts</td><td>1,244</td><td>590</td><td>1,834</td><td>32.2</td><td>78</td><td>22</td><td>100</td><td>22.0</td></tr><tr><td>Planar Geom.</td><td>780</td><td>338</td><td>1,118</td><td>30.2</td><td>65</td><td>35</td><td>100</td><td>35.0</td></tr><tr><td>3D Shapes</td><td>344</td><td>111</td><td>455</td><td>24.4</td><td>75</td><td>25</td><td>100</td><td>25.0</td></tr><tr><td>Graph Struct.</td><td>1,787</td><td>892</td><td>2,679</td><td>33.3</td><td>72</td><td>28</td><td>100</td><td>28.0</td></tr><tr><td>Chemistry</td><td>250</td><td>116</td><td>366</td><td>31.7</td><td>74</td><td>26</td><td>100</td><td>26.0</td></tr><tr><td>Circuit</td><td>480</td><td>214</td><td>694</td><td>30.8</td><td>70</td><td>30</td><td>100</td><td>30.0</td></tr><tr><td>Total</td><td>4,885</td><td>2,261</td><td>7,146</td><td>31.6</td><td>434</td><td>166</td><td>600</td><td>27.7</td></tr></table>

OI-NUM (numerical answer, 5,013 questions), OI-TERM (domain-specific term or label, 1,779 questions), and OI-LIST (list of elements, 354 questions). This distribution reflects that the majority of questions require numerical computation (e.g., calculating area, degree, or resistance values), followed by identification of domain-specific entities.

(5) Mini split. The mini split is a class-balanced subset of the full dataset, containing exactly 50 diagrams per domain (300 total) with 1,500 evaluation instances (300 D2C-P + 600 D2C-E + 600 DQA). The per-domain task breakdowns of the mini split are shown in Tables B.2 and B.3.

## B.2 Task Generation Pipeline

Figure 2 illustrates the overall benchmark construction pipeline. Herein, we detail the task-specific generation procedure:

(1) Diagram-to-Code Parsing (D2C-P). D2C-P uses fixed task templates instantiated directly from the original diagrams. The task question asks the model to convert the diagram image into compilable TikZ code, conditioned on a provided preamble (document class, required packages, and libraries). The original TikZ source code serves as the ground-truth answer. No agentic generation is required for this task.

(2) Diagram-to-Code Editing (D2C-E). D2C-E employs an agentic pipeline with two main components: a generation agent (Gemini-3 Flash) and a verifier agent comprising two judge models (GPT-5.2 and Gemini-3 Pro) for cross-validation. The pipeline proceeds as follows:

• Template predefinition. We manually design a pool of editing task templates across four dimensions (text, color, scope, layout) for each of the six domains (Table 3). Each template contains placeholders for diagram-specific elements (e.g., labels, colors, vertices, and edges).

• Task selection. Given a diagram and its TikZ source code, the generation agent selects two editing tasks from the template pool of the corresponding domain, constrained to cover distinct editing dimensions to ensure diversity.

• Answer generation. The generation agent instantiates the selected templates with specific diagram elements and generates the corresponding modified TikZ code. Multi-engine compilation (pdflatex→lualatex→xelatex) is performed to verify executability.

• Agent verification. The verifier agents evaluate generated results against predefined criteria: all edits must be visually observable and conform to the selected editing dimension, and unafected elements must remain unchanged. A case is accepted only when both verifier agents agree.

• Iterative refinement. Failed cases, together with verifier feedback, are returned to the generation agent for regeneration, with a maximum of three attempts.

• Human verification. All accepted instances are manually reviewed by 13 graduate students. Each annotator’s assigned instances are cross-validated by another annotator to ensure language clarity, logical consistency, and answer correctness.

(3) Diagram Question Answering (DQA). The DQA generation pipeline follows the same agentic workflow as D2C-E; only the task templates difer. The generation agent produces descriptive and reasoning questions based on the diagram, its TikZ source code, and domain-specific templates encoding symbol semantics and domain knowledge (§B.6). Template placeholders are instantiated using diagram elements extracted from the source code. For example, in organic chemistry, functional groups (e.g., COOH or OH) are extracted to generate questions such as “which functional group is present in this molecule?”. The verifier agents then assess each (diagram, question, answer) triple for logical consistency with both visual evidence and domain knowledge; regeneration with verifier feedback is allowed for up to three attempts, and only validated QA pairs are retained. Human verification follows the same cross-validation protocol as D2C-E.

Each diagram yields five evaluation instances (1 D2C-P + 2 D2C-E + 2 DQA). The template-guided selection enforces dimensional diversity: D2C-E instances cover distinct editing dimensions per diagram, and DQA questions span both descriptive and reasoning types. All templates and rubric designs are detailed in §B.4 and §B.6.

## B.3 D2C-P Examples

Figure B.1 presents representative D2C-P case studies across diagram domains. Each case shows the input diagram, the task prompt, the ground-truth TikZ code, and the rendered output.

## Diagram-to-Code Parsing (D2C-P)

## Charts Case 1

Q: Recreate this diagram using LAT X. Provide the full, executable code. Cycle 2 marks between successive plots, then colors

![](images/bd120f03e121095d73333916708ab6fafe39a29d45dba40040c2d694838dbc5d.jpg)

## A:

% [1 packages: pgfplots]   
\begin{tikzpicture}   
\begin{axis}[   
title={Cycle 2 marks between successive plots, then colors}, cycle multi list={   
color list\nextlist   
[2 of]mark list   
},   
...   
\addplot {x−8};   
\addplot {x−9};   
\addplot {x−10};   
\addplot {x−11};   
\end{axis}   
\end{tikzpicture}

Rendered Image:

![](images/1a7e8f466a5b679817da6929479d67fc00a7bc0e7debdb427e85bb7926c5f07b.jpg)

## Charts Case 2

Q: Recreate this diagram using LAT<sub>E</sub>X. Provide the full, executable code.

![](images/fb8abf16be8ff4f598fa8246e1199dd38a914388182c3c744de61144097eef2e.jpg)

## A:

% [1 packages: tikz]   
\begin{tikzpicture}[pie chart,slice type={A}{blu},slice type={B}{rosso},slice type ={C}{gray},pie values/.style={font=\small},scale=2]   
\pie{Title}{50/A,25/B,25/C}\legend[shift={(0,−1cm)}]{{A (50\,\%)}/A,{B (25\,\%) }/B,{C (25\,\%)}/C}   
\end{tikzpicture}

Rendered Image:

![](images/111f48831c58282ea2ba343f0b271adcc64f7d94cf05cca8dc99ae9b9c78d540.jpg)

## Graph Structures Case 1

Q: Recreate this diagram using LAT X. Provide the full, executable code.

![](images/95c876ffb81b24b11a89be9c814e5bebf24aec9385bc9c524a9d054e7e48e5f7.jpg)

## Graph Structures Case 2

Q: Recreate this diagram using LAT<sub>E</sub>X. Provide the full, executable code.

![](images/3686b52f61497beffe2b2cb93826cc4a3d611c62493d62fe828732cde28eb9e8.jpg)

## Planar Geometry Case 1

Q: Recreate this diagram using LAT X. Provide the full, executable code.

![](images/7ab79d20403359e91093186c14c66825fd7de60ad22326180ab4038cb3220134.jpg)

## A:

\begin{tikzpicture}[>=triangle 60] \draw[thick,fill=black] (0,0) circle (1.5mm);   
\draw[thick,fill=black] (5,0) circle (1.5mm);   
\draw[thick,fill=black] (2.5,−3.5) circle (1.5mm);   
\node [left] at (−0.15, 0.2) {\$C\$};   
\node [right] at (5.2,0.2) {\$S\$};   
\node [below] at (2.5,−3.7) {\$T\$}; \draw[−>−=.5,thick](2.5,−3.5) to [bend left=20](0,0);   
\draw[−>−=.5,thick](0,0) to [bend left=20](2.5,−3.5);   
\draw[yscale=−1,yshift=8.55cm,−>−=.5,thick](2.5,−4.27) circle (0.8cm);   
\draw[−>−=.5,thick](2.5,−3.5) to [bend left=20](5,0);   
\draw[−>−=.5,thick](5,0) to [bend left=20](2.5,−3.5);   
\end{tikzpicture}

## A:

% [5 packages: fontenc, inputenc, lmodern...]   
\begin{tikzpicture}   
[vertex/.style={minimum size=2pt,fill,draw,circle},   
open/.style={fill=none},   
sibling distance=1.5cm,level distance=.75cm,   
every fit/.style={ellipse,draw,inner sep=−2pt},   
leaf/.style={label={[name=#1]below:\$#1\$}},auto]   
\node [vertex] (root) {} child { node [vertex,leaf=c] {} }   
child { node [vertex,leaf=f] {} } } }   
edge from parent node {\$\rho\$} };   
\node [fit=(d) (e) (b) (b's parent),label=above left:\$F^{(b,R)}\$] {}; \node [fit=(c) (f) (f's parent),label=above right:\$F^{(c,R)}\$] {}; \end{tikzpicture}

## A:

% [3 packages: amsmath,amssymb, xcolor, tkz−euclide]   
\begin{tikzpicture}[scale=.75]   
\tkzDefPoints{1/5/A,0/0/B,7/0/C}   
\tkzDefSpcTriangle[orthic](A,B,C){H\_A,H\_B,H\_C}   
\tkzDefTriangleCenter[ortho](B,C,A)   
\tkzGetPoint{H}   
\tkzDrawSegments[new](A,H\_A B,H\_B C,H\_C)   
\tkzMarkRightAngles[fill=gray!20,opacity=.5](A,H\_A,C B,H\_B,A C,H\_C,A)   
\tkzLabelPoints[left](B)   
\tkzLabelPoints[above](A)   
\tkzLabelPoints[new](H\_A)   
\tkzLabelPoints[new,above left](H\_C)   
\tkzLabelPoints[new,above right](H\_B,H)   
\end{tikzpicture}

Rendered Image:

![](images/f1d7833d0821ec8f05ea98307810de88e3d6d4495478089fbaceb74b6d81078c.jpg)

## Rendered Image:

![](images/69318f01f12449beebb6dbb94232cedb6f3906f656d7e4bd8cf2c3db36b08a5d.jpg)

## Rendered Image:

![](images/49261ab24eaa0f07b1677773a85eab6c5f5c2c7fe9a26db7de5fc2b9f2d678a9.jpg)

## Planar Geometry Case 2

Q: Recreate this diagram using LAT X. Provide the full, executable code.

![](images/bb141b3c509cf488810991afbd477f4f71948aa08cc315a3c53b97a3473d1237.jpg)

## 3D Shapes Case 1

Q: Recreate this diagram using LAT<sub>E</sub>X. Provide the full, executable code.

![](images/23bf6070055e140d8c0c913bacb1f4a2c786f444e9d8f126ac094d141e351b5c.jpg)

## 3D Shapes Case 2

Q: Recreate this diagram using LAT<sub>E</sub>X. Provide the full, executable code.

![](images/be805c7fc9d61ca6f03d498ab77ef1c655433bf1284088f32eb263bb95df7a23.jpg)

## A:

% [1 packages: tikz]   
\begin{tikzpicture}[scale=3]   
\clip(−2,−0.2)rectangle(2,0.8);   
\draw[step=.5cm,gray,very thin](−1.4,−1.4)grid(1.4,1.4);   
\filldraw[fill=green!20,draw=green!50!black](0,0)−−(3mm,0mm)arc[start angle=0, end angle=30,radius=3mm]−−cycle;   
\draw[−>](−1.5,0)−−(1.5,0)coordinate(x axis);   
\draw[−>](0,−1.5)−−(0,1.5)coordinate(y axis);   
\draw(0,0)circle[radius=1cm];   
\path[name path=sloped line](0,0)−−(30:1.5cm);   
\draw[name intersections={of=upward line and sloped line,by=t}][very thick,orange ](1,0)−−node[right=1pt,fill=white]{\$\displaystyle\tan\alpha=\frac{\sin\alpha }{\color{blue}\cos\alpha}\$}(t);   
\draw(0,0)−−(t);   
\foreach\x/\xtext in{−1,−0.5/−\frac{1}{2},1}\draw(\x cm,1pt)−−(\x cm,−1pt)node[ anchor=north,fill=white]{\$\xtext\$};   
\foreach\y/\ytext in{−1,−0.5/−\frac{1}{2},0.5/\frac{1}{2},1}\draw(1pt,\y cm)−−(−1 pt,\y cm)node[anchor=east,fill=white]{\$\ytext\$};   
\end{tikzpicture}

## A:

% [2 packages: tikz,bm,pgfplots, tikz−3dplot]   
\tdplotsetmaincoords{80}{140}   
\begin{tikzpicture}[scale=2,tdplot\_main\_coords]   
\tikzset{dot/.style={circle,fill,minimum size=4.5pt,inner sep=0pt,outer sep=0pt}   
,hemispherebehind/.style={ball color=gray!20!white,opacity=0.3},hemispherefront/. style={ball color=gray!65!white,opacity=0.3},circlearc/.style={thick,gray !90},circlearchidden/.style={thick,dashed,gray!90},equator/.style={thick, black},diameter/.style={thick,black,stealth−stealth,shorten <=5pt,shorten >=5pt}}\newcommand{\hemispherefront}{(1cm,0)arc(0:−180:1cm and 1.8 mm)arc(180:0:1cm and −1cm)}   
\newcommand{\hemispherebehind}{(1cm,0)arc(0:−180:1cm and −1.8mm)arc (−180:0:1cm and 1cm)}   
\newcommand{\equator}{(−1,0,0)arc(0:360:−1)}   
\tdplotsetthetaplanecoords{35}\path[tdplot\_rotated\_coords,name path=semicircle ](0,−1,0)arc(90:−90:−1);   
\draw[circlearchidden,intersection segments={of=semicircle and hemisphere, sequence={L2}}];   
\draw[circlearc,intersection segments={of=semicircle and hemisphere,sequence={L 3}}];   
\draw[equator]\equator node[pos=1,label={90:\$E\$}]{};   
\draw[diameter](0,−1,0)node[dot,label={120:\$−x\$}]{}−−(0,1,0)node[dot,label ={30:\$x\$}]{};   
\node at(.75,0,−.5){\$\bm{\mathsf{S}}\_2^−\$};   
\end{tikzpicture} % [1 packages: pgfplots]   
\begin{tikzpicture}   
\begin{axis}[   
colormap access=piecewise constant, colormap/RdPu−6,   
colorbar horizontal,   
1   
\addplot3 [   
surf,   
shader=interp,   
] {x<sub>\*</sub>y};   
\end{axis}   
\end{tikzpicture}

## Rendered Image:

![](images/1dc46f20138aec915ba45a5333b5786dea03159c631891096aeff3ff9e16faec.jpg)

## Rendered Image:

## A:

![](images/01badc123ffd2d443c1bb1ca64036ced9a012e5a6669075656ec6b679c0dadb2.jpg)

## Rendered Image:

![](images/c6f8093d18986c27ae4b87014098f65464bf94c41502938d309822a0f2eeaba2.jpg)

## Circuit Diagrams Case 1

Q: Recreate this diagram using LAT X. Provide the full, executable code.

![](images/c3bd3989c502691dabe8a8f42b0e37fc385f20a0547497e21f75f09ff5cc96c5.jpg)

## Circuit Diagrams Case 2

Q: Recreate this diagram using LAT X. Provide the full, executable code.

![](images/777ffc23a03ceaf257beb4577488ac5aaa9ccb6514eaed958adadddadb4570a7.jpg)

## Chemistry Case 1

Q: Recreate this diagram using LAT<sub>E</sub>X. Provide the full, executable code.

![](images/603cabd7a57edd1c50236d110802a2214676597f25c7af2fe9991fb17c5fd47d.jpg)

## Chemistry Case 2

Q: Recreate this diagram using LAT X. Provide the full, executable code.

![](images/be76dc2ebb6007b7792d67d0c07a3c259e85f637e7f648497f79a8c9dba3293b.jpg)

## A:

% [3 packages: fontenc, inputenc, circuitikz] \begin{circuitikz}   
\draw   
(0,0) node[inst amp ra] (opamp) {}   
(opamp.+) node[left] {\$v\_+\$}   
(opamp.−) node[left] {\$v\_−\$}   
(opamp.out) node[right] {out}   
(opamp.up) node[vcc]{}   
(opamp.down) node[vee] {}   
(opamp.refv down) node[ground]{}   
(opamp.refv up) to[short, −o] ++(0,0.3)   
(opamp.ra−) to[R] (opamp.ra+);   
\end{circuitikz}

## A:

% [4 packages: fontenc, inputenc, lmodern...] \begin{circuitikz}   
\draw   
(0,2) node[and port] (myand1) {}   
(0,0) node[and port] (myand2) {}   
(2,1) node[xnor port] (myxnor) {}   
(myand1.out) −| (myxnor.in 1)   
(myand2.out) −| (myxnor.in 2);   
\end{circuitikz}

## A:

% [1 packages: chemfig]   
\chemfig{H\_3C−C(=[:30]O)−[:−30]OH}   
\vflipnext   
\chemfig{H\_3C−C(=[:30]O)−[:−30]OH}\medskip   
\chemfig{H\_3C−C(=[:30]O)−[:−30]OH}   
\hflipnext   
\chemfig{H\_3C−C(=[:30]O)−[:−30]OH}

## A:

% [3 packages: amsmath,amssymb, xcolor, chemfig] \chemfig{?(−[:190]OH)−[:−50](−[:170]OH)−[:10](−[:−55,0.7]OH) −[:−10](−[6,0.7]OH)−[:130]O−[:190]?(−[:150,0.7]−[2,0.7]OH)}

## Rendered Image:

![](images/08dbab75ee97b2c83384dd05026a92aa1215e342550a4937847a0fbe6777acac.jpg)

## Rendered Image:

![](images/b634eb41f70fb65881e5c544e01c05333c0fc39f41799b65383137e2e3d1ea09.jpg)

## Rendered Image:

![](images/27818a6d32ce36af0035e0b15364a652b75195c3fc4b680ae4ec855ad26f1b06.jpg)

## Rendered Image:

![](images/e684fcf6c1b4ee10ae3b11579af4d10277699c9888ac0ca1375f44bae1894cc7.jpg)  
Figure B.1 Representative D2C-P (Diagram-to-Code Parsing) examples. Given an input diagram, the task requires generating complete, compilable L<sup>A</sup>T<sub>E</sub>X/TikZ code that faithfully reproduces the original.

## B.4 D2C-E Templates

The D2C-E employs a pool of 17 editing task templates across four evaluation dimensions: color (C, 2 templates), text (T, 4 templates), scope (S, 8 templates), and layout (L, 3 templates). Each template specifies a naturallanguage editing instruction with placeholders (in square brackets) that are instantiated with diagram-specific

Table B.4 Eediting template pool of D2C-E. 17 templates span four evaluation dimensions. Placeholders in brackets are instantiated with diagram-specific elements during generation.
<table><tr><td>ID</td><td>Task Name</td><td>Template</td></tr><tr><td>Color</td><td></td><td></td></tr><tr><td>C1</td><td>Element Color Change</td><td>Change the [color_ attr] of [target_ element] to [new_ color].</td></tr><tr><td>C2</td><td>Text Color Change</td><td>Change the text color of [target_ text] to [new_ color].</td></tr><tr><td>Text</td><td></td><td></td></tr><tr><td>T1</td><td>Label/Annotation Rename</td><td>Rename the label [old_ text] to [new_text].</td></tr><tr><td>T2</td><td>Data Value Modification</td><td>Change the data value at [position] from [old_value] to [new_- value].</td></tr><tr><td>T3</td><td>Component Parameter Mod.</td><td>Change the parameter of [component] from [old_param] to [new_-</td></tr><tr><td>T4</td><td>Title/Legend Text Mod.</td><td>param]. Change the [text_ type] text from [old_text] to [new_text].</td></tr><tr><td>Scope</td><td></td><td></td></tr><tr><td>S1</td><td>Element Position Move</td><td>Move [target_ element] to [new_position].</td></tr><tr><td>S2</td><td>Rotation/Mirror Transform</td><td>Rotate [target] by [degrees]° [direction]. / Mirror [target] along</td></tr><tr><td>S3</td><td>Axis Range Scaling</td><td>the [axis] axis. Change the [axis] axis range from [old_range] to [new_range].</td></tr><tr><td>S4</td><td>Element Position Swap</td><td>Swap the positions of [element_ A] and [element_ B].</td></tr><tr><td>S5</td><td>Element Deletion</td><td>Remove [target_ element] from the diagram.</td></tr><tr><td>S6</td><td>Element Addition</td><td>Add [new_ element] to the diagram.</td></tr><tr><td>S7</td><td>Node Shape Replacement</td><td>Change the shape of [target_node] from [old_ shape] to [new_-</td></tr><tr><td>S8</td><td></td><td>shape].</td></tr><tr><td>Layout</td><td>Component Type Replacement</td><td>Replace [old_ component] with [new_ component_ type].</td></tr><tr><td>L1</td><td></td><td></td></tr><tr><td>L2</td><td>Structure Type Conversion</td><td>Convert [current_ type] to [target_ type].</td></tr><tr><td></td><td>Conditional Filtering</td><td>Filter the data to only show [condition].</td></tr><tr><td>L3</td><td>Reference Element Addition</td><td>Add [reference_ element] to the diagram.</td></tr></table>

elements during the agentic pipeline (§B.2). Table B.4 presents the complete template pool, and Table B.5 shows per-domain applicability.

## B.5 D2C-E Examples

Figure B.2 presents representative D2C-E cases. Each case shows the original diagram, an editing instruction, the modified TikZ code, and the rendered output after editing.

Table B.5 Editing dimensions for each type of diagram. ✓ indicates the dimension applies to that domain.
<table><tr><td>Task</td><td>Charts</td><td>P. Geom.</td><td>3D</td><td>Graph</td><td>Chem.</td><td>Circuit</td><td>#</td></tr><tr><td>C1</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>6</td></tr><tr><td>C2</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>6</td></tr><tr><td>T1</td><td>√</td><td>V</td><td>J</td><td>√</td><td>V</td><td>√</td><td>6</td></tr><tr><td>T2</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td>1</td></tr><tr><td>T3</td><td></td><td></td><td></td><td></td><td></td><td>√</td><td>1</td></tr><tr><td>T4</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td>1</td></tr><tr><td>S1</td><td>√</td><td>√</td><td></td><td>√</td><td></td><td>√</td><td>4</td></tr><tr><td>S2</td><td></td><td>√</td><td>V</td><td>√</td><td>V</td><td></td><td>4</td></tr><tr><td>S3</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td>1</td></tr><tr><td>S4</td><td></td><td>V</td><td></td><td>√</td><td></td><td>√</td><td>3</td></tr><tr><td>S5</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>6</td></tr><tr><td>S6</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>6</td></tr><tr><td>S7</td><td></td><td></td><td></td><td>√</td><td></td><td></td><td>1</td></tr><tr><td>S8</td><td></td><td></td><td></td><td></td><td></td><td>√</td><td>1</td></tr><tr><td>L1</td><td>√</td><td></td><td></td><td>√</td><td></td><td>√</td><td>3</td></tr><tr><td>L2</td><td>√</td><td></td><td></td><td>√</td><td></td><td>√</td><td>3</td></tr><tr><td>L3</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td>2</td></tr><tr><td>Per domain</td><td>12</td><td>9</td><td>6</td><td>11</td><td>6</td><td>11</td><td></td></tr></table>

## Diagram-to-Code Editing (D2C-E)

## Charts Case 1

Q: Convert this grouped bar chart into a line chart, where each series in the legend is represented by a separate line connecting the data points for each agent.

![](images/e9f0220cfb0ac626a73ec3ed7004c43d0427cef9c5a6003204ac629244d838ff.jpg)

## A:

\begin{axis}[width=\textwidth, height=0.7\textwidth, xlabel={Agent}, ylabel={ Score (\% out of total possible points)}, symbolic x coords={random, LLM Only, TaskGen}, xtick=data, nodes near coords, ymin=0, ymax=110, legend style={at={(0.5,−0.25)}, anchor=north, legend columns=2}, enlarge x limits ={abs=1.5cm}, ylabel near ticks, xticklabel style={rotate=45, anchor=east}, title={Score for each Agent across 6 environments}]

\addplot coordinates {(random,42) (LLM Only,93) (TaskGen,96)};   
\addplot coordinates {(random,0) (LLM Only,20) (TaskGen,30)};   
\addplot coordinates {(random,0) (LLM Only, 20) (TaskGen,30)};   
\addplot coordinates {(random,0) (LLM Only,57) (TaskGen,88)};   
\addplot coordinates {(random,0) (LLM Only,0) (TaskGen,0)};   
\addplot coordinates {(random,0) (LLM Only,0) (TaskGen,0)};

\legend{Detailed Dense (commands provided), Brief Sparse (commands provided ), None Sparse (commands provided), Detailed Dense (commands not provided), Brief Sparse (commands not provided), None Sparse (commands not provided)}

Rendered Image:  
![](images/c4119cd9eb65979c4891839c3a1834bb470206c578e75186daf57155dfc39741.jpg)  
Detailed Dense (commands provided) None Sparse (eommands prováded) → Detailed Dense [commands not provided Brief Sparse (coeumands pat peovided)

## Charts Case 2

Q: Change the data value at the third data point of the Benchmark System from (150, 18.73) to (150,

![](images/a7ea5e7d272bd2921d149e2cd428bc4ae9e38f32e5a0709e32da97acf6b18b69.jpg)

## Graph Structures Case 1

Q: Convert the current hexagonal cycle layout into a linear vertical layout where the nodes are connected in sequence from 6 at the top to 1 at the bottom.

![](images/0eacfd61296f688fb0f3371f5ab713b8e83bdff47fc12f58a7a1a13a656329db.jpg)

## Graph Structures Case 2

Q: Change the shape of node 6 from a circle to a rectangle.

![](images/ad0e8130a39b4d6b1aa410e7a6ba093ddcd7bb7a2ec27d691189f42e00e2deb3.jpg)

## Planar Geometry Case 1

Q: Add a vertical diameter passing through the center point O to the diagram.

![](images/71e56cbd70d0938082f120fe487f4d795f7c1398c58ad95d72e5a8acf97c8870.jpg)

## A:

\begin{axis}[xlabel={Road segment length (m)},ylabel={DQJL establishment time (s)},xmin=80,xmax=250,ymin=0,ymax=35,xtick ={80,100,150,200,250},ytick={5, 10, 15, 20, 25, 30, 35},legend pos=south east,ymajorgrids=true,xmajorgrids=true,grid style=dashed]\addplot[color= blue]coordinates{(80, 8.91)(100, 10.13)(150, 17.03)(200, 22.92)(250, 26.71) };

\addplot coordinates{(80, 10.2)(100, 12.63)(150, 22.00)(200, 25.79)(250, 31.43)

\legend{Centralized System, Benchmark System}

\end{tikzpicture}

## A:

% [3 packages: amsmath, amssymb, tikz,color] \begin{tikzpicture}[scale=0.65] \node [draw, circle] (1) at (0,0) {\$1\$}; \node [draw, circle] (2) at (0,1.5) {\$2\$}; \node [draw, circle] (3) at (0,3) {\$3\$}; \node [draw, circle] (4) at (0,4.5) {\$4\$}; \node [draw, circle] (5) at (0,6) {\$5\$}; \node [draw, circle] (6) at (0,7.5) {\$6\$}; \draw [thick] (6)−−(5)−−(4)−−(3)−−(2)−−(1); \end{tikzpicture}

## A:

% [3 packages: amsmath, amssymb, tikz,color]   
\begin{tikzpicture}[scale=0.65] \node [draw, circle] (1) at (−1.35,0) {\$1\$}; \node [draw, circle] (2) at (1.35,0) {\$2\$}; \node [draw, circle] (3) at (2.5,2) {\$3\$}; \node [draw, circle] (4) at (1.35,4) {\$4\$}; \node [draw, circle] (5) at (−1.35,4) {\$5\$}; \node [draw, rectangle] (6) at (−2.5,2) {\$6\$}; \draw [thick] (1)−−(2)−−(3)−−(4)−−(5)−−(6)−−(1);   
\end{tikzpicture}

## A:

% [1 packages: lmodern,tikz]   
\begin{tikzpicture}[scale=.3] \draw(0,0)circle(4.01cm); \draw(−1,3.87)−−(−2.83,−2.83); \draw(4.01,0)−−(−4.01,0); \draw(0,4.01)−−(0,−4.01); \draw[dash pattern=on 5pt off 5pt](−4.01,0)−−(−1,3.87); \draw[dash pattern=on 5pt off 5pt](4.01,0)−−(−2.83,−2.83);

\fill(4.01,0)circle(2pt)node[blue,right]{\$z^ \$}; \fill(−4.01,0)circle(1.5pt)node[blue,left]{\$z\$}; \fill(−2.05,0)circle(3pt); \node[above left]at(−1.7,−.2){\$y\$}; \fill(0,0)circle(4pt)node[above]{\$O\$}; \end{tikzpicture}

## Rendered Image:

![](images/0fe5b16cb01c8063ae8fd5f0dcef46c2e6b814f6470276df9340b76f9daced67.jpg)

## Rendered Image:

![](images/0c4091dc6d51143362303054e6265b4bfc59d9cafa1150efc145eb196ef47350.jpg)

## Rendered Image:

![](images/7a988d31dd2f4b5aa6dda206932397e03b1ccbdad3d039a0699f2f090549bf18.jpg)

## Rendered Image:

![](images/6a4f51d166596b2d0b4c283d79d377d19a7edf6e5f5048cde5ad74d7a64dc930.jpg)

Rendered Image:

## Planar Geometry Case 2

Q: Change the stroke color of the red path connecting the tangents and the circle center to blue.

![](images/5770dcd6ce005f42f0b09af6c9e6a4dac27089258e9be9d1b1e4522a9f778498.jpg)

## 3D Shapes Case 1

Q: Rotate the 3D surface plot by 90 degrees around the vertical axis.

![](images/7830b4424a874f9d9dcb1075c8574b6cc8eef7bca9c7a425f4b6e7a1a9dec7a5.jpg)

## 3D Shapes Case 2

Q: Change the interior colormap of the 3D surface from ’hot’ to ’plasma’.

![](images/232519ef0875b7dd759146074c5cec99104f9fb6533018eff08e8b946275d7c6.jpg)

## Circuit Diagrams Case 1

Q: Replace the resistor connected between the ra- and ra+ pins with a capacitor.

![](images/9fa6bcc1fa0d9ad1c1f0cea1878ea0e752225356a10ffffad37493b144be33cf.jpg)

## A:

% [1 packages: tikz]   
\begin{tikzpicture} \draw[help lines] (0,0) grid (3,2); \coordinate (a) at (3,2); \node [circle,draw] (c) at (1,1) [minimum size=40pt] {\$c\$}; \draw[blue] (a) −− (tangent cs:node=c,point={(a)},solution=1) −− (c.center) −− (tangent cs:node=c,point={(a)},solution=2) −− cycle;   
\end{tikzpicture}

## A:

% [1 packages: pgfplots]   
\begin{tikzpicture} \begin{axis}[colormap/viridis, view={115}{30}] \addplot3 [ surf, shader=flat, draw=black, samples=10, domain=0:1, ] {x^2<sub>\*</sub>y}; \end{axis}   
\end{tikzpicture}

## A:

% [1 packages: pgfplots] \begin{tikzpicture} \begin{axis}[ hide axis, xlabel=\$x\$,ylabel=\$y\$, mesh/interior colormap name=plasma, colormap/blackwhite, ] \addplot3 [domain=−1.5:1.5,surf] {−exp(−x^2−y^2)}; \end{axis}   
\end{tikzpicture}

## A:

% [3 packages: fontenc, inputenc, circuitikz] \begin{circuitikz} \draw (0,0) node[inst amp ra] (opamp) {} (opamp.+) node[left] {\$v\_+\$} (opamp.−) node[left] {\$v\_−\$} (opamp.out) node[right] {out} (opamp.up) node[vcc]{} (opamp.down) node[vee] {} (opamp.refv down) node[ground]{} (opamp.refv up) to[short, −o] ++(0,0.3) (opamp.ra−) to[C] (opamp.ra+); \end{circuitikz}

![](images/6d52d9133d67fa8cdc9415ae4179d11fdc2f9effec947b8ed5f7eefdf03dbc0a.jpg)

## Rendered Image:

![](images/a3d9681e1b0183f20f06dcfc67cee3b5829725e1e7ed26ba518c87c3881f7df3.jpg)

![](images/6bfc2af8a61050b585b91ed5700e710095d0e0329dcbf5961a8fa2a2f3607d37.jpg)

## Rendered Image:

![](images/f6364b5b8139a1409ccff2360e2428bb9962ae6b515d6dbce1dcd02987d455f9.jpg)

![](images/7fbe2ea683c549ac0ee6c14700b33ed94188869e05096b8de2f17eeb890900df.jpg)

![](images/489365b55b6765dec858dbdb79840aac9f3ec2a65147fa7df76a7bf5961fd4c4.jpg)  
Figure B.2 Representative D2C-E (Diagram-to-Code Editing) examples. Given an input diagram and an editing instruction, the task requires modifying the TikZ code to produce the desired change while preserving unafected elements.

## B.6 DQA Templates

The DQA uses 60 manually designed question templates across the six diagram domains, including descriptive questions (23 templates) that assess the model’s ability to identify domain-specific symbols and extract information; standard reasoning questions (18 templates) that require numerical computation with domainspecific formulas; and what-if reasoning questions (19 templates) that require predicting answers conditioned on hypothetical element modifications. Each template encodes the relevant domain knowledge. Table B.6 summarizes the per-domain distribution. We instruct models with the expected output format at inference, and each question specifies one of three output formats:

• OI-NUM: A single numerical value with units if applicable (e.g., “100 Ω”, “90 deg”, “3.14”). Fallback: NOT\_- PRESENT if information is absent from the diagram.

• OI-TERM: A standard scientific term or exact label (e.g., “Parallelogram”, “Diode”, “Hydroxyl”). Fallback: NOT\_PRESENT.

• OI-LIST: A list of strings sorted alphabetically (e.g., [‘Node A’, ‘Node C’]). Fallback: empty list [].

Table B.6 Question template and instruction counts for DQA across six diagram domains. Templates cover descriptive (Desc.) and reasoning (Reas.) types, where reasoning includes standard (Std.) and what-if sub-types.

<table><tr><td rowspan="2">Domain</td><td colspan="3">Template Count</td><td rowspan="2"></td><td colspan="3">Output Instruction</td></tr><tr><td>Desc.</td><td>Std./Reas.</td><td>What-if/Reas.</td><td>Total NUM</td><td>TERM</td><td>LIST</td></tr><tr><td>Charts</td><td>4</td><td>4</td><td>3</td><td>11</td><td>9</td><td>2</td><td>0</td></tr><tr><td>Planar Geom.</td><td>5</td><td>3</td><td>3</td><td>11</td><td>6</td><td>5</td><td>0</td></tr><tr><td>3D Shapes</td><td>2</td><td>5</td><td>4</td><td>11</td><td>7</td><td>4</td><td>0</td></tr><tr><td>Graph Struct.</td><td>4</td><td>2</td><td>3</td><td>9</td><td>8</td><td>0</td><td>1</td></tr><tr><td>Chemistry</td><td>4</td><td>2</td><td>3</td><td>9</td><td>4</td><td>5</td><td>0</td></tr><tr><td>Circuit</td><td>4</td><td>2</td><td>3</td><td>9</td><td>5</td><td>4</td><td>0</td></tr><tr><td>Total</td><td>23</td><td>18</td><td>19</td><td>60</td><td>39</td><td>20</td><td>1</td></tr></table>

Question templates. (1) Table B.7 presents 41 question templates: 23 descriptive and 18 standard reasoning. Descriptive templates assess symbol identification (e.g., recognizing a zigzag as a resistor, a right-angle square $\mathrm { a s 9 0 ^ { \circ } } )$ , element counting (e.g., vertices, atoms, faces), and pattern recognition (e.g., data trends, series/parallel topology). Standard reasoning templates require applying domain-specific formulas $( { \underline { { \mathrm { e . g . } } } } , V = I R$ for circuits, $A = { \frac { 1 } { 2 } } b h$ for triangles, interior angle sum $= ( n { - } 2 ) \times 1 8 0 ^ { \circ } )$ or performing visual reasoning $( \underline { { \mathrm { e . g . } } }$ , shortest path length, cross-section shape identification); (2) Table B.9 presents 19 what-if templates for the reasoning questions. Each poses a hypothetical modification to the diagram (e.g., changing a data value, removing an element, scaling a dimension) and asks about the consequence. These modifications are deliberately aligned with D2C-E editing operations: for instance, a what-if question about removing a node (aligned with scope task S5) tests whether the model can reason about connectivity changes, while a question about modifying a resistance value (aligned with text task T3) tests circuit analysis under parameter changes.

Domain knowledge. Each template encodes domain-specific knowledge required for correct answers. The knowledge spans: graph theory concepts (degree, in-/out-degree, connected components, shortest path) for Graph Structures; statistical measures (mean, median, range, monotonicity) for Charts; Euclidean geometry (shape classification, perimeter, area, interior angle sum) for Planar Geometry; circuit analysis (Ohm’s law $V { = } I R ,$ series/parallel resistance, power) for Circuits; solid geometry (face/vertex counting, volume, surface area, cross-sections) for 3D Shapes; and organic chemistry conventions (implicit hydrogens from valence rules, bond types, functional group identification, molecular weight) for Chemistry. We summarize the domain knowledge per diagram type in Table 2a.

Table B.7 Question templates in DQA (Part 1: Graph Structures, Charts, Planar Geometry). “D” = descriptive; “R” = standard reasoning.
<table><tr><td>ID</td><td>Type</td><td>Name</td><td>Question Template</td><td>0l</td></tr><tr><td colspan="5">Graph Structures</td></tr><tr><td>Q1</td><td>D</td><td>Degree Query</td><td>What is the degree of the node labeled [label]?</td><td>NUM</td></tr><tr><td>Q2</td><td>D</td><td>In-degree Query</td><td>What is the in-degree of the node labeled [label]?</td><td>NUM</td></tr><tr><td>Q3</td><td>D</td><td>Out-degree Query</td><td>What is the out-degree of the node labeled [label]?</td><td>NUM</td></tr><tr><td>Q4</td><td>D</td><td>Neighbors Query</td><td>List all nodes that are direct neighbors of [label].</td><td>LIST</td></tr><tr><td>Q5</td><td>R</td><td>Path Length</td><td>What is the length of the shortest path from [node_A] to [node_- B]?</td><td>NUM</td></tr><tr><td>Q6</td><td>R</td><td>Tree Height</td><td>What is the height of this tree?</td><td>NUM</td></tr><tr><td colspan="5">Charts</td></tr><tr><td>Q1</td><td>D</td><td>Max/Min Query</td><td>Which data point/category has the maximum (or minimum) value?</td><td>TERM</td></tr><tr><td>Q2</td><td>R</td><td>Mean Query</td><td>What is the mean (average) value of all visible data points?</td><td>NUM</td></tr><tr><td>Q3</td><td>R</td><td>Median Query</td><td>What is the median value of the dataset?</td><td>NUM</td></tr><tr><td>Q4</td><td>R</td><td>Range Query</td><td>What is the range (max - min) of the dataset?</td><td>NUM</td></tr><tr><td>Q5</td><td>D</td><td>Monotonicity</td><td>In the interval from [start] to [end], what trend does the data</td><td>TERM</td></tr><tr><td>Q6</td><td>R</td><td>Extrema Location</td><td>exhibit? At approximately what x-coordinate does the function reach its</td><td>NUM</td></tr><tr><td>Q7</td><td>D</td><td>Heatmap Value</td><td>minimum? According to the color scale, what is the approximate value at cell</td><td>NUM</td></tr><tr><td>Q8</td><td>D</td><td>Polar Plot</td><td>([row], [col])? At angle θ = [angle]°, what is the radial distance r?</td><td>NUM</td></tr><tr><td colspan="5">Planar Geometry</td></tr><tr><td>Q1</td><td>D</td><td>Right Angle Symbol</td><td>What angle measure does the small square symbol at [location] indicate?</td><td>NUM</td></tr><tr><td>Q2</td><td>D</td><td>Equal Length Marks</td><td>Sides [A] and [B] have identical tick marks. What does this indicate?</td><td>TERM</td></tr><tr><td>Q3</td><td>D</td><td>Dashed Line</td><td>What does the dashed line in this geometric diagram typically</td><td>TERM</td></tr><tr><td>Q4</td><td>D</td><td>Triangle Type</td><td>represent? Based on the marked information, what type of triangle is [label]?</td><td>TERM</td></tr><tr><td>Q5</td><td>D</td><td>Quadrilateral Type</td><td>Based on the marked properties, what type of quadrilateral is [label]?</td><td>TERM</td></tr><tr><td>Q6</td><td>R</td><td>Perimeter</td><td>If side [A] = [v1] and side [B] = [v2], what is the perimeter of</td><td>NUM</td></tr><tr><td>Q7</td><td>R</td><td>Area</td><td>rectangle [rect]? If base [b] = [v1] and height [h] = [v2], what is the area of triangle</td><td>NUM</td></tr><tr><td>Q8</td><td>R</td><td>Interior Angle Sum</td><td>[tri]? What is the sum of interior angles of this [n]-sided polygon?</td><td>NUM</td></tr></table>

Table B.8 Question templates in DQA (Part 2: Circuit Diagrams, 3D Shapes, Chemistry). “D” = descriptive; “R” = standard reasoning.
<table><tr><td>ID</td><td>Type</td><td>Name</td><td>Question Template</td><td>01</td></tr><tr><td colspan="5">Circuit Diagrams</td></tr><tr><td>Q1</td><td>D</td><td>Resistor ID</td><td>What component is represented by the zigzag symbol labeled [label]?</td><td>TERM</td></tr><tr><td>Q2</td><td>D</td><td>Capacitor ID</td><td>What component is represented by the two parallel lines labeled [label]?</td><td>TERM</td></tr><tr><td>Q3</td><td>D</td><td>Diode ID</td><td>What component is represented by the triangle-with-line symbol labeled [label]?</td><td>TERM</td></tr><tr><td>Q4</td><td>D</td><td>Series/Parallel</td><td>Are components [comp_ A] and [comp_B] connected in series or</td><td>TERM</td></tr><tr><td>Q5</td><td>R</td><td>Total Resistance</td><td>parallel? If [R1] = [v1]Ω and [R2] = [v2]Ω are in series, what is the total</td><td>NUM</td></tr><tr><td>Q6</td><td>R</td><td>Ohm&#x27;s Law</td><td>resistance? If [R] has resistance [val] Ω and current [I] A flows through it, what is the voltage?</td><td>NUM</td></tr><tr><td colspan="5">3D Shapes</td></tr><tr><td>Q1</td><td>D</td><td>Solid Identification</td><td>What is the name of this 3D solid?</td><td>TERM</td></tr><tr><td>Q2</td><td>R</td><td>Face Count</td><td>How many faces does this solid have?</td><td>NUM</td></tr><tr><td>Q3</td><td>R</td><td>Vertex Count</td><td>How many vertices does this solid have?</td><td>NUM</td></tr><tr><td>Q4</td><td>D</td><td>Hidden Edges</td><td>What do the dashed lines in this 3D diagram represent?</td><td>TERM</td></tr><tr><td>Q5</td><td>R</td><td>Cross-section</td><td>If this [solid] is cut horizontally, what is the shape of the cross- section?</td><td>TERM</td></tr><tr><td>Q6</td><td>R</td><td>Volume</td><td>If the cylinder has radius [r] and height [h], what is its volume? (Use π = 3.14)</td><td>NUM</td></tr><tr><td>Q7</td><td>R</td><td>Surface Area</td><td>If the cube has edge length [e], what is its surface area?</td><td>NUM</td></tr><tr><td colspan="5">Chemistry</td></tr><tr><td>Q1 Q2</td><td>D</td><td>Unlabeled Vertex Bond Type</td><td>What atom does the unlabeled vertex at [location] represent? What type of bond connects atom [A] to atom [B]?</td><td>TERM</td></tr><tr><td>Q3</td><td>D</td><td>Implicit Hydrogens</td><td>How many implicit hydrogen atoms are bonded to the carbon at</td><td>TERM</td></tr><tr><td></td><td>R</td><td></td><td>position [pos]?</td><td>NUM</td></tr><tr><td>Q4</td><td>D</td><td>Functional Group</td><td>What functional group is present at [location]?</td><td>TERM</td></tr><tr><td></td><td>D</td><td>Carbon Count</td><td>How many carbon atoms are in this molecule (including labeled and unlabeled)?</td><td>NUM</td></tr><tr><td>Q6</td><td>R</td><td>Molecular Formula</td><td>What is the molecular formula of this compound?</td><td>TERM</td></tr></table>

Table B.9 DQA what-if question templates. Each template specifies a hypothetical modification and asks about its consequence.
<table><tr><td>ID</td><td>Name</td><td>Question Template</td><td>0I</td></tr><tr><td colspan="4">Graph Structures</td></tr><tr><td>W1</td><td>Node Removal</td><td>Remove node [target] and all its edges. How many connected compo- nents remain?</td><td>NUM</td></tr><tr><td>W2</td><td>Edge Addition</td><td>Add a new edge between [node_ A] and [node_B]. What is the new degree of [node_ A]?</td><td>NUM</td></tr><tr><td>W3</td><td>Edge Weight Mod.</td><td>Change the edge weight from [A] to [B] from [old] to [new]. What is the new total weight of the path from [start] to [end]?</td><td>NUM</td></tr><tr><td colspan="4">Charts</td></tr><tr><td>W1</td><td>Value Modification</td><td>Change data point [label] from [old] to [new]. What is the new mean of the dataset?</td><td>NUM</td></tr><tr><td>W2</td><td>Data Removal</td><td>Remove data point [target]. What is the new maximum value?</td><td>NUM</td></tr><tr><td>W3</td><td>Uniform Shift</td><td>Increase all data values by [δ]. What is the new range of the dataset?</td><td>NUM</td></tr><tr><td colspan="4">Planar Geometry</td></tr><tr><td>W1</td><td>Side Scaling</td><td>Scale side [label] by a factor of [k]. What is the new perimeter?</td><td>NUM</td></tr><tr><td>W2</td><td>Uniform Scaling</td><td>Scale all sides of the [shape] by a factor of [k]. By what factor does the area increase?</td><td>NUM</td></tr><tr><td>W3</td><td>Angle Modification</td><td>Change angle [A] from [old]° to 90°. What type of triangle does [tri] become?</td><td>TERM</td></tr><tr><td colspan="4">Circuit Diagrams</td></tr><tr><td>W1</td><td>Resistance Mod.</td><td>Change [R] from [old] to [new] Ω. If voltage is [V] V, what is the new current?</td><td>NUM</td></tr><tr><td>W2</td><td>Short Circuit</td><td>Short-circuit [R] (0Ω). What is the current through [R&#x27;] = [val] Ω at [V]V?</td><td>NUM</td></tr><tr><td>W3</td><td>Parallel Addition</td><td>Add [val]Ω in parallel with [R] (also [val] Ω). What is the equivalent resistance?</td><td>NUM</td></tr><tr><td colspan="4">3D Shapes</td></tr><tr><td>W1</td><td>Radius Scaling</td><td>Scale the cylinder radius by factor [k], keeping height constant. By what factor does the volume increase?</td><td>NUM</td></tr><tr><td>W2</td><td>Uniform Scaling</td><td>Scale all dimensions of the rectangular prism by factor [k]. By what factor does the surface area increase?</td><td>NUM</td></tr><tr><td>W3</td><td>Cone Slicing</td><td>Remove the top half of the cone by slicing at mid-height. What is the remaining solid called?</td><td>TERM</td></tr><tr><td>W4</td><td>Stacking</td><td>Stack an identical [shape] on top. What is the total volume of the combined structure?</td><td>NUM</td></tr><tr><td colspan="4">Chemistry</td></tr><tr><td>W1</td><td>Bond Type Change</td><td>Change the bond between [A] and [B] from single to double. How many implicit H on [A]?</td><td>NUM</td></tr><tr><td>W2</td><td>Substituent Repl.</td><td>Replace [old_ group] at [pos] with [new_ group]. What is the change in molecular weight?</td><td>NUM</td></tr><tr><td>W3</td><td>Oxygen Insertion</td><td>Insert an O atom between [A] and [B] to form [A]-O-[B]. What functional group is this?</td><td>TERM</td></tr></table>

## B.7 DQA Descriptive Question Examples

Figure B.3 presents representative DQA descriptive question examples. These questions assess the model’s ability to identify domain-specific symbols and extract information directly from the diagram.

## Descriptive Questions in Diagram Question Answering

## Charts Case 1

![](images/e92ede2acaaab9792ba18e53fd2cd89ab9dfbe087d4f4948d3d42ba42564eb41.jpg)

Q: Which category and bar style (by color/pattern) has the maximum value?

A: green with horizontal lines

## Charts Case 2

![](images/1f8871885a440d06841686a3b6d8622a38e932e2d4ae2adab6d4b26f0c8c68d6.jpg)

Q: In the interval from 0 to 1, what trend does the data exhibit?

A: increasing

## Graph Structures Case 1

![](images/d9e3dc5f4013a0f3646b57548c89c3f93f5bccd86dbe5eac1cbe2ae7a07ef07d.jpg)

A: 3

Q: What is the degree of the white node?

## Graph Structures Case 2

![](images/8e9a4fab581aaf74085f49546e661bb30e16c10c407c766a9120cc8ccbd3aa9c.jpg)

A: a, c

Q: List all nodes that are direct neighbors of b.

## Planar Geometry Case 1

![](images/01fe7450ddc48ef67f3f0bd46372b45af7bf1aacfd2f5fd8fbec8dbbf0136a43.jpg)

A: 90 deg

Q: What angle measure does the small square symbol at vertex B indicate?

## Planar Geometry Case 2

![](images/c1f14ce8f2af76d8eddac20d2371aad7a28c0ca20c005fddd82fc02b18b06e99.jpg)

Q: Based on the marked information, what type of triangle is formed by the segments labeled cos α, sin α, and the radius of the circle?

A: right triangle

## 3D Shapes Case 1

![](images/cdc61d773e72b6efd4002c1cd5618344044edf3b91876f4c0a41fe4f007c826b.jpg)

Q: What is the name of this 3D solid?

A: cube

## 3D Shapes Case 2

![](images/4c9d9e4927460b2ce256bbc69f6c5e64c1d2f81fdf7dde56b6a7d5e2862c312a.jpg)

Q: What do the dashed lines in this 3D diagram represent?

A: hidden edges

## Circuit Diagrams Case 1

![](images/f75e669cb232e8156d57550de07a737efe921329bbe49823afc8d29bc1ba8e4d.jpg)

A: resistor

Q: What type of component is represented by the zigzag symbol labeled R1?

## Circuit Diagrams Case 2

![](images/8711be22240591009bc4968238013efe0d38ff640a11fc30b7a9fa94537c5bb4.jpg)

Q: Are the resistor and the component labeled M connected in series or parallel?

A: series

Chemistry Case 1

![](images/309ee91dabcfbaa3fccfb88e35fb2a9f9420db9f795deeef0007f1ab9d92da2e.jpg)

A: single bond

Q: What type of bond connects atom A to atom B?

Charts Case 2  
![](images/6d6cd522fb2b71ac85323ac5b64395de9d4973bbadfb5e599b9c0b6bb17ba00c.jpg)  
Figure B.3 Representative DQA descriptive question examples. Each case shows a diagram and a question that requires direct observation and information extraction from the visual content.

Q: What functional group is present at the terminal position on the right?

A: carboxyl group

## B.8 DQA Reasoning Question Examples

Figure B.4 presents representative DQA reasoning question examples. These questions require numerical computation with domain-specific formulas or predicting outcomes of hypothetical modifications.

## Reasoning Questions in Diagram Question Answering

## Charts Case 1

![](images/027b2114a9ed7326dd92541437676409f6a3724cf99e6569eab43652bc9ea190.jpg)

A: 200

Q: What is the range (difference between maximum and minimum) of the vertical y-axis values shown in the dataset?

![](images/88af7e11be35aed8e3a6a5f6917273eaf5ac20e35b428d20f87274fcc26f6add.jpg)

A: 7

Q: Consider the following modification to the dataset in the top-right circular radar chart: Change the value of data point ’Money’ from 5 to 10. (Type: T4) What is the new mean of the entire dataset?

## Graph Structures Case 1

![](images/3142c4de7727e9c6ed9d5c9d551a808ad3633c1d53654658c371313e8d671747.jpg)

A: 3

Q: What is the length of the shortest path from Neutral to Digital [On]?

## Graph Structures Case 2

![](images/74c43b6d2c3bb69b613ccc0b1608f070131941a9182123699cccaa13a9abd2dc.jpg)

Q: Consider the following modification: Remove the node 4 and all its connected edges. (Type: T5) How many connected components remain in the graph?

Planar Geometry Case 1  
![](images/5b19c712ec7cbb6807be2df2ecd6711009d431e3b9ac7f865b5306b26d5bff82.jpg)

Q: In the diagram on the right, if the vertical base of triangle T\~2 along the right edge is b = 10 units and its horizontal height from the vertex is 5 units, what is the area of triangle T\~2? A: 12.5

Planar Geometry Case 2  
![](images/65f812df28ca7c651438f8e6e9c7a5599d25a34e8516be181b259fb5219b8e29.jpg)

Q: Consider the following modification: Scale the radius of the circle (currently 2 units) by a factor of 2. (Type: T2) What is the new perimeter of the full circle?

3D Shapes Case 1  
![](images/d66857aa16165fa16a3c430dedd64f10cf23a0b12e13046e396a1efcfe66932c.jpg)  
A: 6.28

Q: If the cylinder that bounds this helix has radius 1 and height 2, what is its volume? (Use pi = 3.14)

3D Shapes Case 2  
![](images/b02d269f6ca42be39216684dac670d548d8b0e4837845050bac51ad5bd11afa4.jpg)

Q: If this hemisphere is cut horizontally, what is the shape of the cross-section? A: circle

## Circuit Diagrams Case 1

![](images/35027f346f34b75db1a4fa49e814861af7032d7030a2b0a6b8a16fd0e86e9b73.jpg)  
A: 5V

Q: If resistor R\_0 has resistance 1000 Ohms and current I\_0 = 0.005 A flows through it, what is the voltage across it?

## Circuit Diagrams Case 2

![](images/1a3dbbb642f672f6e3990de982f53b50de9efbcd316f4cbe8c39d129e0e678c8.jpg)

A: 50 Ohm

Q: Consider the following modification: Add a resistor of 100 Ohms in parallel with the existing resistor R1 (also 100 Ohms). (Type: T9) What is the equivalent resistance of this parallel combination?

![](images/1d288a4a5c0fef1aa6eba628134480be5c9fd828206e15aa8f167b38b6d6ca28.jpg)  
Figure B.4 Representative DQA reasoning question examples. Each case shows a diagram and a question that requires domain-specific reasoning, computation, or hypothetical analysis beyond direct observation.

## C Evaluation Methodology

## C.1 Semantic Object Model (SOM) Pipeline

Our object-based metric evaluates whether the model correctly perceives basic objects that the code draws. We parse both generated and ground-truth TikZ code into a set of graphical objects—their type, text content, color, and bounding box—and compute F1 scores for each dimension. The extraction proceeds through three stages: semantic injection, compilation, and DOM-based extraction.

Stage 1: Semantic injection via semantic\_spy.sty. Before compilation, we preprocess the TikZ source by injecting a custom LAT<sub>E</sub>X style file, semantic\_spy.sty, as the last loaded package in the preamble. This package hooks into TikZ, pgfplots, and CircuiTikZ commands via \special{dvisvgm:raw ...} directives, which embed semantic XML tags directly into the DVI output. Specifically, it installs the following hooks:

• TikZ core hooks. Every \node is wrapped in <gclass="tikz-node">. The wrapper stores data-id, data-shape, and optionally data-text. Similarly, every tikzpicture and scope environment is wrapped with corresponding semantic group tags.

• pgfplots data hooks. Every axis environment is wrapped with <gclass="pgf-axis">. Within each \addplot, scatter marker hooks inject <gclass="data-point"> tags with original coordinates (data-x, data-y) and parent-series identifiers. Each plot series is wrapped with <gclass="pgf-series">.

• CircuiTikZ component hooks. Every bipole component is wrapped with <gclass="circuit-component">. The wrapper carries a unique component identifier.

The preprocessor also handles code standardization: wrapping fragments in standalone documents, detecting and injecting missing packages/libraries, fixing common compilation issues (duplicate packages, xcolor option clashes, pgfplots compatibility version downgrades), and checking DVI compatibility. If the code is incompatible with DVI mode (e.g., uses fontspec or xeCJK), semantic injection is skipped and the pipeline falls back to geometry-only extraction.

Stage 2: Compilation and SVG conversion. The preprocessed code is compiled to DVI format using the latex engine, which preserves the \special directives containing the semantic tags. If DVI compilation fails, the pipeline falls back to PDF-based compilation with multi-engine fallback (pdflatex → lualatex → xelatex), which still produces geometry-only SVG. The DVI (or PDF) output is then converted to SVG using dvisvgm with parameters --font-format=svg (to preserve text elements), --precision=6, and --zoom=1 (for 1:1 coordinate mapping). For DVI input, dvisvgm faithfully translates the injected \special tags into SVG group elements with the corresponding CSS classes and data attributes. For PDF input, dvisvgm--pdf produces geometric SVG without semantic annotations.

Stage 3: DOM-based SOM extraction. The generated SVG is parsed using an lxml-based DOM parser. The extractor traverses the SVG tree recursively and classifies each element into one of the following categories: • Semantic elements: identified by CSS class attributes injected by semantic\_spy.sty. These include tikz-node (Node), tikz-path (Path), pgf-axis (Axis), pgf-series (DataSeries), data-point (DataPoint), and circuit-component (Component).

Table C.1 Primary SOM element types extracted per domain. All domains share basic Node, Path, and Text types; domain-specific types arise from specialized packages.
<table><tr><td>Domain</td><td>Primary TikZ Pkg.</td><td>Key SOM Element Types</td></tr><tr><td>Charts</td><td>pgfplots</td><td>AXIS, DATASERIES, DATAPOINT, NODE (legend), TEXT (tick labels)</td></tr><tr><td>Graph Struct.</td><td>tikz</td><td>NoDE (circle, rectangle, diamond), PATH (open/closed), TEXT (labels)</td></tr><tr><td>Planar Geom.</td><td>tikz/tkz-euclide</td><td>NoDE (point markers), PATH (edges, arcs), TEXT (vertex labels, annotations)</td></tr><tr><td>Circuit</td><td>circuitikz</td><td>COMPONENT (R, C, L, V, etc.), NoDE (junctions), TEXT (value labels)</td></tr><tr><td>3D Shapes</td><td>pgfplots/tikz</td><td>AXIS, PATH (surfaces, wireframes), NODE (labels), FILL/FILLDRAW</td></tr><tr><td>Chemistry</td><td>chemfig</td><td>NoDE (atoms, functional groups), PATH (bonds), TEXT (atom labels)</td></tr></table>

• Native SVG shapes: identified by SVG tag names (rect, circle, ellipse, path, text, etc.), mapped to corresponding geometric types.

For each element, the extractor records four attributes:

• Type. A fine-grained type key that combines the coarse element type with subtype attributes, e.g., node: circle, node:rectangle, path:closed, path:open, and component:R (resistor). Structural container types such as picture, scope, and axis are excluded from type-level evaluation because they would inflate the score.

• Text. Text content is extracted with the following priority: (1) the data-text attribute from semantic injection; (2) the metadata child element; (3) glyph reconstruction from <use> elements (see below); and (4) fallback to <text> or <tspan> children. For DataPoint elements, the $( x , y )$ data values are formatted as text for matching.

• Color. Extracted from fill and stroke attributes for path elements, and from fill for text elements. Colors specified as none or transparent are excluded.

• BBox. Axis-aligned bounding boxes are computed from SVG geometry: directly from coordinate attributes for basic shapes (rect, circle, ellipse), from path data using svgpathtools for <path> elements, and recursively aggregated for group elements. Cumulative SVG transforms (translate, scale, matrix) are tracked through the DOM tree.

Glyph text reconstruction. A key challenge is that dvisvgm converts text into glyph paths stored in <defs> and referenced by <use> elements. Their IDs follow the pattern g{font\_id}-{char\_code}. To recover text content, we collect all <use> elements within a group, sort them by x-position (left to right), decode the character codes from glyph IDs, and concatenate the characters. For example, <usehre $\mathtt { f } = \ " \# \mathtt { g } 0 - 6 5 \ " / >$ corresponds to character code 65 (“A”). This reconstruction handles both single-character and multi-character text groups, including colored text whose parent <g> carries a fill attribute.

Per-domain element mapping. The SOM pipeline maps diagrams from diferent TikZ packages into a unified element schema. Table C.1 summarizes the primary element types extracted per domain.

## C.2 Evaluation Metrics

Object-based metric computation. Formally, given TikZ source code c in domain $d ,$ a domain extractor $\mathcal { E } _ { d }$ maps code to attribute sets:

$$
\mathcal { E } _ { d } ( c ) = \big \{ \mathcal { T } _ { d } ( c ) , \mathcal { X } _ { d } ( c ) , \mathcal { K } _ { d } ( c ) , \mathcal { B } _ { d } ( c ) \big \} .\tag{1}
$$

Here, $\mathcal { T } _ { d } ( c )$ is a multiset of primitive types; $\mathcal { X } _ { d } ( c )$ is a multiset of textual elements; $\mathcal { K } _ { d } ( c )$ is a multiset of normalized color specifications; and $B _ { d } ( c )$ is a set of axis-aligned bounding boxes $b _ { i } = ( x , y , w , h )$ , with $( x , y )$ denoting the center coordinates and w and h representing the width and height.

Primitive F1 Score. Given predicted code cˆ and ground-truth code $c ,$ we extract 4 graphic objects via $\mathcal { E } _ { d }$ . Let $M _ { * }$ denote the number of correct matches (defined below), $| \hat { S } |$ the predicted count, and |S| the ground-truth count. The per-dimension F1 is defined as follows, where the subscript \* is replaced by each instance in {type, text, color, bbox}.

$$
\mathrm { P } _ { * } = \frac { M _ { * } } { | \hat { S } | } , \quad \mathrm { R } _ { * } = \frac { M _ { * } } { | S | } , \quad \mathrm { F } 1 _ { * } = \frac { 2 \mathrm { P } _ { * } \mathrm { R } _ { * } } { \mathrm { P } _ { * } + \mathrm { R } _ { * } } .\tag{2}
$$

The four dimensions difer only in how $M _ { * }$ is computed: (1) Count-based multiset matching for type. Let $\hat { n } ( t )$ and n(t) be the counts of type $t \in \mathcal { T } _ { d }$ in predicted and ground-truth code:

$$
\begin{array} { r } { M _ { t y p e } = \sum _ { t \in \mathcal { T } _ { d } } \operatorname* { m i n } ( \hat { n } ( t ) , n ( t ) ) . } \end{array}\tag{3}
$$

(2) Greedy one-to-one exact string matching for text. Each ground-truth text element matches at most one predicted element by exact string equality: $M _ { t e x t } = | { \mathrm { m a t c h e d ~ p a i r s } } | .$ (3) Permutation-based optimal assignment for color [14]. Colors are grouped by associated type of element. Within each group g, the optimal permutation π of predicted colors is selected using CIE2000 perceptual similarity:

$$
\begin{array} { r } { M _ { c o l o r } = \sum _ { g } \operatorname* { m a x } _ { \pi } \sum _ { i } \sin \left( \pi ( \hat { k } _ { i } ^ { g } ) , k _ { i } ^ { g } \right) , \quad \sin ( \hat { k } , k ) = \operatorname* { m a x } \left( 0 , \ 1 - \frac { \Delta E _ { 0 0 } ( \hat { k } , k ) } { 1 0 0 } \right) . } \end{array}\tag{4}
$$

(4) Greedy one-to-one IoU matching for BBox. For each ground-truth box b (iterated in order), we search the remaining unmatched predicted boxes and select the first <sup>ˆ</sup>b whose Intersection-over-Union (IoU) exceeds a threshold $\tau { = } 0 . 3 \colon$ IoU $\begin{array} { r } { \mathbf { \hat { \rho } } ( \hat { b } , b ) = \frac { | \hat { b } \cap b | } { | \hat { b } \cup b | } \geq \tau . } \end{array}$ The matched predicted box is then removed from the candidate pool. The final matching count is $M _ { b b o x } =$ |matched pairs|.

The primary perception-level metric is the average across the four objects:

$$
\begin{array} { r } { \mathrm { F } 1 _ { a v g } = \frac { 1 } { 4 } \left( \mathrm { F } 1 _ { t y p e } + \mathrm { F } 1 _ { t e x t } + \mathrm { F } 1 _ { c o l o r } + \mathrm { F } 1 _ { b b o x } \right) . } \end{array}\tag{5}
$$

Code-based metric computation. We use CrystalBLEU [36] to measure token-level similarity between generated and ground-truth TikZ code. CrystalBLEU extends standard BLEU-4 by filtering out trivially shared n-grams—high-frequency, low-information tokens such as $\{ , \} , \ [ , ] , \ , \ , \ , =$ , \begin, and \end—that would inflate scores without reflecting meaningful code similarity. Tokenization is performed using the Pygments TexLexer, which correctly segments LAT X control sequences, and tokens of type Whitespace and Comment are discarded.

For the editing task, CrystalBLEU is also decomposed into preserve-only and edit-only variants. The edit-only CrystalBLEU performs multiset subtraction of source n-grams from both GT and generated n-grams before computing BLEU, isolating the contribution of newly introduced code. The preserve-only variant keeps only n-grams that intersect with the source (Counter intersection), measuring how faithfully the model retains unchanged code portions.

Image-based metric computation. We compile both generated and ground-truth TikZ code into PNG images (300 DPI, cropped whitespace) and compare them using four established image similarity metrics. The rendering pipeline uses latexmk with multi-engine fallback (pdflatex → lualatex → xelatex), followed by PDF cropping (via pdfCropMargins) and pdftoppm conversion to PNG. All images are resized to $2 5 6 \times 2 5 6$ pixels with white background before metric computation.

• SSIM [32]. Structural Similarity Index Measure, computed via torchmetrics with data range normalized to [0, 1]. Compares brightness, contrast, and structural patterns between two images (higher is better).

• CLIP Score [33]. Cosine similarity between CLIP ViT-B/32 image embeddings of the generated and groundtruth images. Captures high-level semantic similarity in a shared vision-language embedding space (higher is better).

• LPIPS [34]. Learned Perceptual Image Patch Similarity, computed using an AlexNet backbone. Measures perceptual distance using deep features that correlate with human perceptual judgments (lower is better).

• FID [35]. Fréchet Inception Distance, a distribution-level metric computed over all generated and groundtruth images within each model’s evaluation set using torchmetrics with Inception v3 features (2048-dim). Requires $\geq 2$ images per set (lower is better).

In the main results (Table 5), we report two composite image-level scores: SC (the average of SSIM and CLIP, higher is better) and FL (the average of FID and LPIPS, lower is better).

Edge cases. When both the predicted and ground-truth sets are empty for a given dimension $( \underline { { \mathrm { e . g . } } } , \underline { { \cdot } } ,$ neither code contains any text elements), we define F1 = 1.0 (perfect match), since the model correctly produces no elements of that type. When the color group contains $\leq 6$ elements, we use exhaustive permutation search; for larger groups $( > 6$ elements), we switch to greedy approximation to maintain computational tractability.

Preserve/edit splits in D2C-E. For the diagram-to-code editing task, a single overall score is insuficient: an edit typically changes only a small portion of the diagram, so a model that ignores the editing instruction and simply reproduces the original diagram could already score high. We therefore split evaluation into two complementary parts:

(1) Preserve-only evaluation measures how well the model retains elements that should remain unchanged after editing. For each dimension, we keep only elements present in both the generated code and the original source code (intersection), then compute F1 score against the ground-truth preserved elements. Formally, for a given dimension:

• GT preserve set: GT elements that match source elements.

• Gen preserve set: Generated elements that match source elements.

• F1 score is computed between the GT preserve set and the Gen preserve set.

(2) Edit-only evaluation isolates the quality of the actual edits. For each dimension, elements common between the source and the ground-truth are subtracted from both the ground-truth and the generated code before computing F1 score. This eliminates the contribution of unchanged elements and focuses exclusively on what the editing instruction asks to change. Formally:

• Common set: Elements present in both source and GT (computed via greedy matching).

• GT edit set: GT elements after removing the common set.

• Gen edit set: Generated elements after removing the common set.

• F1 score is computed between the GT edit set and the Gen edit set.

We apply the same procedure for object- and code-based metric computation on both preserve and edit splits.

## C.3 DQA LLM-as-a-Judge Details

We evaluate DQA responses through a two-stage pipeline: rule-based matching followed by LLM-as-a-judge fallback for ambiguous cases.

Stage 1: Answer extraction. The model’s raw output is processed to extract the final answer. Models are instructed to return JSON {"answer": "...", "reasoning": "..."}, but actual outputs may contain markdown code blocks, extra text, or malformed JSON. The extractor tries four strategies in order: (1) direct JSON parsing; (2) extraction from markdown “‘json ... “‘ code blocks; (3) finding the outermost {...} braces; (4) regex fallback for "answer": "..." patterns. Thinking tags (<think>...</think>) are stripped before extraction.

Stage 2: Rule-based matching. The extracted answer is compared against the ground truth using type-specific rules based on the output instruction (OI) type:

• OI-NUM (5,013 questions, 70.2%): Numeric comparison with absolute tolerance $( \leq 1 0 ^ { - 6 } )$ or relative tolerance (≤ 1%), with unit normalization (e.g., “degrees” = “deg” = “<sup>◦</sup>”; “amps” = “A”).

• OI-TERM (1,779 questions, 24.9%): Case-insensitive exact string match after normalization (removing quotes, collapsing whitespace, stripping trailing punctuation).

• OI-LIST (354 questions, 5.0%): Order-independent set comparison. Lists are parsed from JSON arrays or CSV format, and items are normalized before set equality check.

Each rule produces a confidence score. When confidence is high (≥ 0.85), the rule verdict is accepted directly. When confidence is low (e.g., partial containment for terms, or unparseable candidate), the case is escalated to Stage 3.

Stage 3: LLM-as-a-judge fallback. For cases where rule-based matching yields low confidence, we use an LLM judge. Given a question, the ground-truth answer, and the model’s response, we prompt the judge LLM to classify the response as Correct, Incorrect, or Unattempted. The judge prompt is type-aware: it includes OI-specific guidelines (e.g., for OI-NUM, small rounding diferences are acceptable; for OI-LIST, extra or missing items make the answer incorrect). The judge outputs a structured two-line response: (1) an evaluation explaining the reasoning, and (2) a label. The final binary score is 1 for Correct and 0 otherwise. The full judge prompt template is shown below:

Table C.2 Human agreement validation of the DQA LLM judge. κ<sub>J-H</sub>: Cohen’s Kappa between the LLM judge and human majority vote. κ<sub>H-H</sub>: Fleiss’ Kappa among three human annotators. Acc: percentage agreement between judge and human majority. <sup>†</sup>Stage 2 κ values are deflated by extreme label skew (99%+ correct); raw accuracy better reflects the near-perfect agreement on this subset.
<table><tr><td>Subset</td><td>N</td><td>KJ-H</td><td>KH-H</td><td>Acc (%)</td></tr><tr><td>All</td><td>200</td><td>0.937</td><td>0.915</td><td>97.5</td></tr><tr><td>Stage 2 (Rule)</td><td>130</td><td>0.000†</td><td>0.663†</td><td>99.2</td></tr><tr><td>Stage 3 (LLM)</td><td>70</td><td>0.839</td><td>0.805</td><td>94.3</td></tr><tr><td>OI-NUM</td><td>133</td><td>0.961</td><td>0.949</td><td>98.5</td></tr><tr><td>OI-TERM</td><td>53</td><td>0.911</td><td>0.809</td><td>96.2</td></tr><tr><td>OI-LIST</td><td>14</td><td>0.811</td><td>1.000</td><td>92.9</td></tr></table>

## DQA LLM-as-a-Judge Prompt

Role: You are an expert judge specialized in evaluating the correctness of answers to scientific diagram understanding   
questions.   
Task: Classify the model’s response into one of three categories:   
1. Correct: The model answer contains the core information of the ground truth and is semantically consistent.   
2. Incorrect: The model answer contradicts the ground truth.   
3. Unattempted: The model explicitly states it cannot answer, or the response is empty.   
Answer Type: {output\_instruction}   
{OI-specific guidelines}   
Output Format:   
1. Evaluation: [Brief explanation of reasoning]   
2. Label: [Correct / Incorrect / Unattempted]   
Input:   
Question: {question}   
Model Answer: {model\_answer}   
Ground Truth Answer: {ground\_truth\_answer}

## C.4 LLM Judge Human Agreement Validation

To verify the reliability of our LLM-as-a-judge evaluation, we conduct a human agreement study following prior work that validates automated evaluators against independent human annotations and human-majority labels using agreement-based metrics [63].

Sampling. We stratify-sample 200 cases from all 7,146 DQA questions along two dimensions: (1) evaluation stage, 70 cases resolved by the LLM judge (Stage 3) and 130 by rule-based matching (Stage 2), which ensures coverage of the most subjective cases; and (2) output instruction type, 133 OI-NUM, 53 OI-TERM, and 14 OI-LIST, approximately following the corpus proportions. We use GPT-5.2 model outputs as the representative responses for annotation.

Annotation protocol. Three graduate-student annotators independently label each (question, model\_answer, ground\_truth) triple as Correct, Incorrect, or Unattempted, following the same OI-specific grading guidelines used by the LLM judge (§C.3). Annotators are shown the diagram image, question text, ground-truth answer, and model answer, but not the automated judge’s verdict.

Agreement metrics. We report Cohen’s Kappa (κ) and raw accuracy. Specifically, κ<sub>J-H</sub> measures agreement between the LLM judge and the human majority vote, while κ<sub>H-H</sub> measures Fleiss’ Kappa among the three human annotators as a human reference. We additionally report raw accuracy because κ can be deflated under extreme label imbalance, even when agreement is very high [63].

Results. As shown in Table C.2, the LLM judge closely matches human consensus overall, achieving $\kappa _ { \mathrm { J - H } } = 0 . 9 3 7$ with 97.5% accuracy, indicating almost perfect agreement. This is comparable to the human agreement level $\left( \kappa _ { \mathrm { H - H } } = 0 . 9 1 5 \right)$ , suggesting that the automated judge reliably tracks majority human judgments. For the 130 Stage 2 cases, only one judge–human disagreement occurs (99.2% accuracy); the near-zero κ is an artifact of extreme label skew rather than poor agreement. For the 70 Stage 3 cases—the most subjective subset—the judge still achieves $\kappa _ { \mathrm { J - H } } = 0 . 8 3 9$ (94.3% accuracy), close to the corresponding human agreement

Table D.1 Generation configurations of evaluated models. “API” denotes cloud-hosted model endpoints; “vLLM” denotes models deployed with vLLM. T: sampling temperature; $L _ { \mathrm { m a x } } \colon$ maximum output tokens.
<table><tr><td>Model</td><td>Inference</td><td>T</td><td> $L _ { \mathrm { m a x } }$ </td><td>Notes</td></tr><tr><td colspan="5">Closed-Source</td></tr><tr><td>Gemini-3.1 Pro</td><td>API</td><td>1.0</td><td>16,384</td><td rowspan="6">reasoning_effort = medium</td></tr><tr><td>Gemini-3.0 Pro</td><td>API</td><td>1.0</td><td>16,384</td></tr><tr><td>Gemini-3.0 Flash</td><td>API</td><td>1.0</td><td>16,384</td></tr><tr><td>GPT-5.2</td><td>API</td><td>1.0</td><td>16,384</td></tr><tr><td>Claude-4.6 Opus</td><td>API</td><td>1.0</td><td>16,384</td></tr><tr><td>Seed-2.0 Pro</td><td>API</td><td>1.0</td><td>16,384</td></tr><tr><td colspan="5">Open-Source</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>API</td><td>1.0</td><td>16,384</td><td rowspan="6">1×A800</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>API</td><td>1.0</td><td>6,144</td></tr><tr><td>Kimi-K2.5</td><td>API</td><td>1.0</td><td>16,384</td></tr><tr><td>Qwen3-VL-8B</td><td>vLLM</td><td>1.0</td><td>8,192</td></tr><tr><td>InternVL3-38B</td><td>vLLM</td><td>1.0</td><td>8,192</td></tr><tr><td>TikZero+ 10B</td><td>vLLM</td><td>0.8</td><td>4,096</td></tr></table>

$\left( \kappa _ { \mathrm { H - H } } = 0 . 8 0 5 \right)$ . Across output instruction types, agreement is highest for OI-NUM $\left( \kappa _ { \mathrm { J - H } } = 0 . 9 6 1 \right)$ and lowest for OI-LIST $\left( \kappa _ { \mathrm { J - H } } = 0 . 8 1 1 \right)$ , consistent with the greater ambiguity of list-valued answers.

## D Implementation Details

## D.1 Model Configurations

Table D.1 summarizes the generation configurations for all 12 evaluated models. Closed-source models and several large open-source models are accessed via their respective provider APIs. The remaining open-source models are deployed locally or remotely using vLLM (v0.14.0). All models use a sampling temperature of T=1.0 except TikZero+ 10B (T=0.8). The maximum output length $L _ { \mathrm { m a x } }$ is set to 16,384 tokens for API-based models to accommodate long TikZ programs, while locally-served models use smaller limits (4,096–8,192) due to GPU memory constraints.

## D.2 Inference and Compilation

Inference pipeline. All models receive diagram images encoded as base64 data URIs within multimodal messages. For D2C-P and D2C-E, the system prompt instructs the model to output complete LAT X code inside a single “‘latex code block, starting from \documentclass and ending with \end{document}. For DQA, models return structured JSON with answer and reasoning fields. Generated code is extracted from the model response via regex-based markdown code-block parsing. Each inference call is subject to a 300-second timeout; failed calls are retried up to 3 times with exponential backof (2, 4, 8 s delays).

Hardware. All experiments are conducted on a single machine with 8×NVIDIA A800 80 GB GPUs. Opensource models served locally with vLLM each occupy one GPU. Image-level metric computation (SSIM, LPIPS, CLIP) runs on a dedicated GPU to avoid memory contention with inference. The DQA LLM judge (Qwen3-Next-80B-A3B, §C.3) is deployed across 4 GPUs with tensor parallelism via vLLM.

LAT<sub>E</sub>X compilation. Generated TikZ code is compiled into PDF using a multi-engine fallback strategy: pdflatex is attempted first, followed by lualatex and xelatex if earlier engines fail. Each compilation attempt has a 60-second timeout. Compiled PDFs are cropped using pdfCropMargins and converted to 256×256 PNG images (300 DPI) via pdftoppm for image-level metric computation. For object-level F1 evaluation, code is separately compiled to DVI format and converted to SVG for Structured Object Model (SOM) extraction.

Failure handling. When compilation fails or times out, all upward metrics $( \mathrm { F 1 } _ { a v g } .$ , CrystalBLEU, SSIM, CLIP) default to 0, and downward metrics (LPIPS) default to 1.0, ensuring a fixed denominator across all models. Inference failures, including API errors, empty responses, or timeouts after all retries, receive the same penalty scores. This deterministic treatment prevents missing data from inflating any model’s average performance.

## E Evaluation Prompts

We list 16 evaluation settings (S1-S16) in Table 4. Herein, we provide the full system prompts per setting. Specifically, each inference call follows a two-message structure: (1) a system message containing the task prompt (presented below), and (2) a user message containing the diagram image (base64-encoded) together with a task-specific text query (e.g., “Convert this diagram to LaTeX TikZ code” for D2C-P, an editing instruction for D2C-E, or a question for DQA). Placeholders enclosed in braces ({...}) are filled at inference time with instance-specific data: {perception\_data} is a structured JSON list of visual primitives extracted from ground-truth annotations.

## E.1 Foundational Evaluation Prompts

The three foundational settings use system prompts that instruct the model to perform the task directly without additional context or tools.

S1 / S6: D2C-P Direct Coding / D2C-E Direct Editing

You are a LaTeX expert. Follow these rules strictly:   
1. Output the COMPLETE LaTeX code inside a single “‘latex code block.   
2. Start from \documentclass and end with \end{document}.   
3. Do NOT output any text, explanation, or reasoning outside the code block.

S1 and S6 share the same system prompt; the task is distinguished by the user message: S1 sends only the diagram image, while S6 additionally includes an editing textual instruction.

## S12: DQA Direct Answering

You are a scientific diagram analysis expert. Answer the question based on the image and follow the required answer format exactly.

## E.2 Agentic Evaluation Prompts

The agentic settings augment the foundational prompts with one or more capabilities: context utilization (providing structured object data), tool use (TikZ documentation search via MCP), state management (requiring intermediate code generation), and planning (coordinating multiple capabilities). Below, prompts are organized by task.

## E.2.1 Diagram-to-Code Parsing (S2–S5)

## S2: + Objects (Context Utilization)

You are a LaTeX expert specializing in scientific diagrams.

## Visual Perception of This Diagram

The following is a structured analysis of the visual elements in this diagram, extracted from ground truth data. These are the primitive building blocks that compose the diagram — your generated code must faithfully reproduce all of them. {perception\_data}

## Instructions

• The perception data above describes the fundamental visual primitives: their types, colors, text content, and spatial positions.

• Treat these as the complete inventory of what the diagram contains. Every element listed must be present in your output.

• Named colors (e.g., red, blue, gray) are standard color names.

• Output the COMPLETE code inside a single “‘latex code block. Start from \documentclass and end with \end{document}.

## S3: + TikZ Search Tool (Tool Use)

You are a LaTeX expert with access to a TikZ documentation tool.

## Available Tool

• SearchLaTeXKnowledgeBase: Search TikZ/PGF/pgfplots documentation for syntax and examples. Workflow

1. Observe the diagram image carefully

2. If you need specific TikZ syntax or package usage, use the SearchLaTeXKnowledgeBase tool

3. Generate the COMPLETE LaTeX code inside a single “‘latex code block

4. Start from \documentclass and end with \end{document}

## S4: + Model Generated Objects (State Management)

You are a LaTeX expert. Before writing code, analyze the image systematically.

Step 1 — Perception (inside <perception> tags):

1. Element types

2. Text content

3. Colors

4. Spatial layout

Step 2 — Code Generation: Based on your perception, output the COMPLETE LaTeX code inside a single “‘latex code block. Start from \documentclass and end with \end{document}.

## S5: + Objects & TikZ Search Tool (Planning)

You are a LaTeX expert with access to a TikZ documentation tool.

## Visual Perception of This Diagram

The following is a structured analysis of the visual elements in this diagram, extracted from ground truth data. These are the primitive building blocks that compose the diagram — your generated code must faithfully reproduce all of them. {perception\_data}

## Available Tool

• SearchLaTeXKnowledgeBase: Search TikZ/PGF/pgfplots documentation for syntax and examples.

## Instructions

• The perception data above describes the fundamental visual primitives of this diagram. Every element listed must be present in your output.

• Named colors (e.g., red, blue, gray) are standard color names.

• Use the SearchLaTeXKnowledgeBase tool when you need specific syntax or package usage.

• Output the COMPLETE code inside a single “‘latex code block. Start from \documentclass and end with \end{document}.

## E.2.2 Diagram-to-Code Editing (S7–S11)

## S7: + Objects (Context Utilization)

You are a LaTeX expert.

## Visual Perception of This Diagram

The following is a structured analysis of the visual elements in this diagram, extracted from ground truth data. These are the primitive building blocks that compose the diagram.

{perception\_data}

## Instructions

• The perception data above describes the fundamental visual primitives: their types, colors, text content, and spatial positions.

• Use these primitives to understand the diagram’s structure, then apply the user’s modification instruction.

• Output the final modified code inside a single “‘latex code block.

• Start from \documentclass and end with \end{document}.

## S8: + TikZ Search Tool (Tool Use)

You are a LaTeX expert with access to a TikZ documentation tool.

## Available Tool

• SearchLaTeXKnowledgeBase: Search TikZ/PGF/pgfplots documentation for syntax and examples.

## Workflow

1. Observe the diagram image carefully

2. If you need specific TikZ syntax or package usage for the edit task, use the SearchLaTeXKnowledgeBase tool

3. Apply the user’s modification instruction

4. Output the final modified code inside a single “‘latex code block

5. Start from \documentclass and end with \end{document}

## S9: + Required TikZ Codes (State Management)

You are a LaTeX expert. Complete this task in two steps within a single response.

Step 1 — Reconstruct Original Code: Look at the diagram image and generate complete LaTeX code for the original diagram. Output it inside <original\_reconstruction> tag.

Step 2 — Apply Modification: Based on Step 1, apply the user instruction. Output the final modified code inside a single “‘latex code block.

## Rules:

• The “‘latex code block is the final answer.

• It must start from \documentclass and end with \end{document}.

## S10: + Optional TikZ Codes (Planning)

You are a LaTeX expert. You may optionally reconstruct the code before editing.

If you find it helpful, you can first reconstruct the original diagram code inside <original\_reconstruction> tags. This is entirely optional — skip it if you can apply the edit directly.

Then, apply the user’s modification instruction. Output the final modified code inside a single “‘latex code block. Rules:

• The “‘latex code block is the final answer.

• It must start from \documentclass and end with \end{document}.

## S11: + Optional Codes & TikZ Search Tool (Planning)

You are a LaTeX expert with access to a TikZ documentation tool.

• SearchLaTeXKnowledgeBase: Search TikZ/PGF/pgfplots documentation for syntax and examples. Workflow

You may optionally reconstruct the original diagram code inside <original\_reconstruction> tags before editing. This is entirely optional — skip it if you can apply the edit directly.

If you need specific TikZ syntax or package usage, use the SearchLaTeXKnowledgeBase tool.

Then, apply the user’s modification instruction. Output the final modified code inside a single “‘latex code block. Rules:

• The “‘latex code block is the final answer.

• It must start from \documentclass and end with \end{document}.

## E.2.3 Diagram Question Answering (S13–S16)

## S13: + Objects (Context Utilization)

You are a scientific diagram analysis expert.

## Visual Perception of This Diagram

The following is a structured analysis of the visual elements in this diagram, extracted from ground truth data. These are the primitive building blocks that compose the diagram.

{perception\_data}

Instructions

• The perception data above describes the fundamental visual primitives: their types, colors, text content, and spatial positions.

• Use these primitives along with the image to answer the question.

• Follow the answer format specified in the question.

## S14: + Required TikZ Codes (State Management)

You are a scientific diagram analysis expert.

Step 1 — Code Reconstruction: Generate the complete LaTeX code for this diagram. Output it inside <internal\_code\_- representation> tag.

Step 2 — Answer the Question: Using both image and code, answer the question. Follow the answer format specified by the question. Output final answer after </internal\_code\_representation>.

## S15: + Optional TikZ Codes (Planning)

You are a scientific diagram analysis expert.

If you find it helpful, you can first reconstruct the diagram’s LaTeX code inside <internal\_code\_representation> tags to aid your analysis. This is entirely optional — skip it if you can answer directly from the image.

Then, answer the question based on the image (and your code reconstruction if you made one). Follow the answer format specified by the question. Output your final answer after any code reconstruction tags.

## S16: + Optional Codes & TikZ Search Tool (Planning)

You are a scientific diagram analysis expert with access to a TikZ documentation tool.

• SearchLaTeXKnowledgeBase: Search TikZ/PGF/pgfplots documentation for syntax and examples. Workflow

If you find it helpful, you can first reconstruct the diagram’s LaTeX code inside <internal\_code\_representation> tags. This is entirely optional.

![](images/cbffbea47fd691c49e0c32f602a97e22f3f6032b06a202a12ea2605029dafde0.jpg)

Figure F.1 Three stage construction pipeline for the MCP based TikZ documentation server. L<sup>A</sup>T X source files from 19 package manuals are parsed into a unified JSON knowledge base and then deployed as a Mintlify hosted MCP server. During inference, models query the SearchLaTeXKnowledgeBase tool on demand.

Table F.1 Extraction strategies for the six core TikZ packages aligned with Diagram-MMU’s diagram domains.
<table><tr><td>Package</td><td>Domain(s)</td><td>#Files</td><td>Doc. Format</td><td>Extraction Strategy</td></tr><tr><td>tikz-pgf</td><td>Graph, Geom., 3D</td><td>~150</td><td>codeexample</td><td>Recursive scan of PGF manual tree</td></tr><tr><td>pgfplots</td><td>Charts, 3D</td><td>~80</td><td>codeexample</td><td>Recursive file traversal with axis detection</td></tr><tr><td>circuitikz</td><td>Circuit</td><td>4</td><td>LTXexample</td><td>Component catalog + showexpl examples</td></tr><tr><td>tkz-euclide</td><td>Planar Geom.</td><td>32</td><td>tkzexample</td><td>Multi file orchestration across submodules</td></tr><tr><td>chemfig</td><td>Chemistry</td><td>5</td><td>\exemple macro</td><td>Custom delimiter parsing</td></tr><tr><td>tikz-network</td><td>Graph Struct.</td><td>1</td><td>docspec/lstlisting</td><td>Command and option extraction</td></tr></table>

Table F.2 Example SearchLaTeXKnowledgeBase queries for benchmark domains.
<table><tr><td>Domain</td><td>Example Query</td><td>Returned Types</td></tr><tr><td>Charts</td><td>&quot;addplot bar chart stacked&quot;</td><td>Executable examples, command specs</td></tr><tr><td>Graph Struct.</td><td>&quot;tikz node positioning arrow styles&quot;</td><td>Executable examples, command specs</td></tr><tr><td>Planar Geom.</td><td>&quot;tkz-euclide angle bisector mark right angle&quot;</td><td>Executable examples, command specs</td></tr><tr><td>Circuit</td><td>&quot;circuitikz resistor capacitor parallel&quot;</td><td>Component definitions, examples</td></tr><tr><td>3D Shapes</td><td>&quot;pgfplots 3d surface plot axis options&quot;</td><td>Executable examples, command specs</td></tr><tr><td>Chemistry</td><td>&quot;chemfig bond angle double bond ring&quot;</td><td>Executable examples, key value options</td></tr></table>

If you need to understand specific TikZ syntax or package conventions, use the SearchLaTeXKnowledgeBase tool. Then, answer the question based on the image (and your analysis). Follow the answer format specified by the question. Output your final answer after any code reconstruction tags.

## F MCP-Based TikZ Documentation Server

## F.1 Server Architecture and Implementation

As described in §3.5, we build the TikZ search tool as an MCP [27] server to give models on demand access to curated TikZ documentation, avoiding the noise of web search and the ineficiency of loading full manuals.

Pipeline overview. The server is built in three stages: knowledge extraction from oficial LAT<sub>E</sub>X source files, unified knowledge base construction, and Mintlify based MCP deployment, as shown in Figure F.1.

Stage 1: Knowledge extraction. We extract structured knowledge from the oficial LAT<sub>E</sub>X source files of each TikZ package manual. Since packages use diferent documentation formats, we implement package specific extractors to parse examples, commands, options, and component catalogs. In total, the framework processes about 270 source files from 19 packages using seven Python scripts. Table F.1 summarizes the extraction strategy for the six core packages.

Stage 2: Knowledge base construction. All extracted items are normalized into a unified JSON schema. Each item stores its type, package, name, description, relevant code or syntax, and source file. The merged knowledge base contains 8,809 items in total, mainly executable examples and command specifications.

Stage 3: MCP server deployment. The structured knowledge base is converted into 411 Mintlify documentation pages and exposed through an MCP endpoint. The server provides a single tool, SearchLaTeXKnowledgeBase, which models can call during inference to retrieve relevant documentation fragments.

Query interface. The tool accepts a natural language query and optional filters, and returns ranked results such as code examples, command syntax, and usage notes. It supports keyword search, exact phrase search, command lookup, and Boolean operators. Table F.2 shows representative queries across benchmark domains.

Table F.3 Indexed TikZ package documentation. Only the most important statistics are shown.
<table><tr><td>Package</td><td>Benchmark Domain(s)</td><td>Total</td><td>Exec.</td><td>Cmd.</td><td>%</td></tr><tr><td colspan="6">Core Packages</td></tr><tr><td>tikz-pgf</td><td>Graph, Geom., 3D</td><td>3,696</td><td>2,872</td><td>808</td><td>42.0</td></tr><tr><td>pgfplots</td><td>Charts, 3D</td><td>1,472</td><td>1,294</td><td>167</td><td>16.7</td></tr><tr><td>circuitikz</td><td>Circuit</td><td>816</td><td>501</td><td></td><td>9.3</td></tr><tr><td>tkz-euclide</td><td>Planar Geom.</td><td>467</td><td>357</td><td>110</td><td>5.3</td></tr><tr><td>chemfig</td><td>Chemistry</td><td>290</td><td>248</td><td></td><td>3.3</td></tr><tr><td>tikz-network</td><td>Graph Struct.</td><td>71</td><td>53</td><td>18</td><td>0.8</td></tr><tr><td>Core subtotal</td><td></td><td>6,812</td><td>5,325</td><td>1,103</td><td>77.3</td></tr><tr><td colspan="6">Extended Packages</td></tr><tr><td>xspace</td><td>All</td><td>1,107</td><td>107</td><td>1,000</td><td>12.6</td></tr><tr><td>pst-solides3d</td><td>3D Shapes</td><td>262</td><td>262</td><td></td><td>3.0</td></tr><tr><td>fullpage</td><td>All</td><td>193</td><td>7</td><td>186</td><td>2.2</td></tr><tr><td>amscd</td><td>Graph Struct.</td><td>140</td><td>140</td><td></td><td>1.6</td></tr><tr><td>tkz-base</td><td>Planar Geom.</td><td>97</td><td>97</td><td>一</td><td>1.1</td></tr><tr><td>tkz-graph</td><td>Graph Struct.</td><td>92</td><td>92</td><td></td><td>1.0</td></tr><tr><td colspan="6">Supplemental Packages</td></tr><tr><td>tikz-cd</td><td>Graph Struct.</td><td>31</td><td>31</td><td></td><td>0.4</td></tr><tr><td>comment</td><td>All</td><td>25</td><td>25</td><td></td><td>0.3</td></tr><tr><td>tikz-3dplot</td><td>3D Shapes</td><td>21</td><td>21</td><td></td><td>0.2</td></tr><tr><td>tikz-qtree</td><td>Graph Struct.</td><td>15</td><td>15</td><td></td><td>0.2</td></tr><tr><td>soul</td><td>All</td><td>8</td><td>4</td><td>4</td><td>0.1</td></tr><tr><td>forest</td><td>Graph Struct.</td><td>5</td><td>5</td><td></td><td>0.1</td></tr><tr><td>tcolorbox</td><td>All</td><td>1</td><td>1</td><td>一</td><td>0.0</td></tr><tr><td colspan="2">Total</td><td>8,809</td><td>6,132</td><td>2,293</td><td>100</td></tr></table>

Integration with MLLMs. During agentic evaluation, models receive the SearchLaTeXKnowledgeBase tool description in the system prompt and may invoke it as needed. The MCP design is model agnostic, so any MCP compatible client can connect to the same server without model specific tool implementations.

## F.2 Indexed Documentation List

The knowledge base indexes oficial documentation from 19 LAT<sub>E</sub>X packages. The six core packages are directly aligned with the six diagram domains in Diagram-MMU. The remaining packages either map to one of these six domains or provide domain agnostic support that can be used across all benchmark categories. Table F.3 summarizes the indexed packages. For utility packages that are not tied to a single diagram type, we label the benchmark domain as All rather than leaving the field empty.

Core packages. The six core packages directly correspond to the six diagram domains in Diagram-MMU and contribute 77.3% of all indexed items. Among them, tikz-pgf and pgfplots provide the largest share of documentation, while circuitikz, tkz-euclide, chemfig, and tikz-network cover circuit, planar geometry, chemistry, and graph structure respectively.

Extended and supplemental packages. Among the remaining packages, several can still be mapped to benchmark domains. pst-solides3d and tikz-3dplot support 3D Shapes, while amscd, tkz-graph, tikz-cd, tikz-qtree, and forest support Graph Struct. tkz-base is mapped to Planar Geom. The rest, including xspace, fullpage, comment, soul, and tcolorbox, are domain agnostic utility packages, so we label them as All.

Documentation sources. All items are extracted from oficial package documentation only. We do not include third party tutorials, forum posts, or community generated content, so the knowledge base remains

![](images/3391664b3a41b39652077205c4dca8f1d26b3480e4090dfadb5b80e6b70c6b9e.jpg)  
Figure G.1 (Updated) $\mathrm { F 1 _ { a v g } }$ heatmap per diagram type: left panel for D2C-P, middle for preserve-only D2C-E, and right for edit-only D2C-E. Color encodes score magnitude (red→white→green); the dashed line separates closed-source (top) from open-source (bottom) models. 3D shapes is the most challenging domain across all models, while chemistry is relatively the easiest.

authoritative and consistent.

## G Additional Experimental Results

## G.1 Foundational Results Per Diagram Type

We report Diagram-to-Code Parsing (D2C-P) results for 12 models and Diagram-to-Code Editing (D2C-E) results for 11 models, excluding TikZero+10B as it was not trained on editing samples. Figure G.1 shows $\mathrm { F 1 _ { a v g } }$ scores over each diagram domain: left panel for D2C-P, middle for preserve-only D2C-E, and right for edit-only D2C-E. The heatmap uses a diverging color scale from red (low) through white to green (high). Figure G.2 reports DQA accuracy across 6 diagram domains.

For diagram-to-code parsing and editing, all models struggle on 3D Shapes: D2C-P scores drop as low as 11.2% (TikZero+10B), and even the best model stays below 40%, indicating that current MLLMs fail to perceive three-dimensional spatial relationships and translate them into TikZ code. Most models perform well on chemistry diagrams, as the ChemFig package encodes molecular topology explicitly through bond-chain syntax. Graph diagrams are the relatively easiest editing domain for current MLLMs, likely because models easily ground the editing instructions into node–edge topology. Closer inspection of the D2C-E heatmaps shows that preserve-only scores are higher than edit-only scores. For example, in charts, edit-only scores fall 10–20 points below preserve-only, indicating that current models retain untouched elements well but struggle to apply the instructed modifications. Moreover, the gap between open-source and closed-source models is larger on chemistry and circuit diagrams, especially for D2C-P.

## G.2 F1 Scores Per Object

Figure 6 shows F1 score breakdowns for six representative models across three radars: D2C-P (left), D2C-E preserve-only (center), and D2C-E edit-only (right). Each radar plots five axes: $\mathrm { F 1 _ { a v g } , F 1 _ { t y p e } , F 1 _ { b b o x } , F 1 _ { c o l o r } } _ { \mathrm { : } }$ and $\mathrm { F 1 } _ { \mathrm { t e x t } }$ , with axis scales fitted to each radar’s value range.

A consistent finding is that $\mathrm { F 1 } _ { \mathrm { b b o x } }$ is dramatically lower than other object attributes. In D2C-P, all model polygons collapse toward the center on the $\operatorname { F 1 } _ { \mathrm { b b o x } }$ axis. This asymmetry persists in D2C-E preserve-only evaluation and becomes even more extreme in edit-only evaluation, confirming that spatial positioning remains the primary bottleneck across both parsing and editing tasks. Comparing the three panels shows a clear degradation: preserve-only scores are substantially higher than D2C-P on most axes, especially text and color where models benefit from retaining existing code, while edit-only scores drop sharply. Among the six models, Gemini-3.0 Pro achieves the largest polygon area across all three panels, while Qwen3-VL-8B shows the most compact polygons.

Table G.1 reports the per-object F1 scores on D2C-P, code-level (CrystalBLEU) and image-level (SSIM, CLIP, FID, LPIPS) metrics, complementing the aggregated $\mathrm { F 1 _ { a v g } }$ in Table $5 . \ \mathrm { F 1 _ { b b o x } }$ is the lowest across all models, confirming the spatial-positioning bottleneck identified in Figure 6.

Table G.1 Three-level metric results of 12 models on D2C-P. ↓ means lower is better. Bold is best in class; underline is second best.
<table><tr><td></td><td colspan="5">Object-Level</td><td>Code-Level</td><td colspan="4">Image-Level</td></tr><tr><td>Model</td><td> $\mathrm { F 1 _ { t y p e } }$ </td><td> $\mathrm { F 1 } _ { \mathrm { t e x t } }$ </td><td> $\mathrm { F 1 _ { c o l o r } }$ </td><td> $\mathrm { F 1 } _ { \mathrm { b b o x } }$ </td><td> $\mathrm { F 1 _ { a v g } }$ </td><td>CBLEU</td><td>SSIM</td><td>CLIP</td><td>FID↓</td><td>LPIPS↓</td></tr><tr><td>Closed-Source</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini-3.1 Pro</td><td>59.9</td><td>61.7</td><td>61.2</td><td>16.9</td><td>49.94</td><td>32.88</td><td>54.01</td><td>70.64</td><td>7.66</td><td>43.15</td></tr><tr><td>Gemini-3.0 Pro</td><td>66.8</td><td>69.2</td><td>69.9</td><td>12.7</td><td>54.64</td><td>32.76</td><td>62.13</td><td>82.75</td><td>8.88</td><td>35.14</td></tr><tr><td>Gemini-3.0 Flash</td><td>63.3</td><td>66.1</td><td>65.3</td><td>10.1</td><td>51.19</td><td>30.67</td><td>60.45</td><td>81.63</td><td>9.21</td><td>36.59</td></tr><tr><td>GPT-5.2</td><td>62.7</td><td>68.4</td><td>66.2</td><td>8.0</td><td>51.35</td><td>28.81</td><td>62.66</td><td>88.29</td><td>12.74</td><td>37.53</td></tr><tr><td>Claude-4.6 Opus</td><td>62.7</td><td>71.1</td><td>66.0</td><td>9.2</td><td>52.23</td><td>31.27</td><td>61.11</td><td>86.36</td><td>13.79</td><td>42.63</td></tr><tr><td>Seed-2.0 Pro</td><td>59.8</td><td>62.4</td><td>60.3</td><td>7.8</td><td>47.57</td><td>32.56</td><td>60.14</td><td>84.93</td><td>14.66</td><td>44.30</td></tr><tr><td>Open-Source</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-397B-A17B</td><td>61.8</td><td>66.2</td><td>63.6</td><td>14.0</td><td>51.38</td><td>32.92</td><td>61.86</td><td>85.83</td><td>11.06</td><td>39.28</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>66.2</td><td>70.9</td><td>66.8</td><td>16.6</td><td>55.11</td><td>37.92</td><td>61.87</td><td>87.93</td><td>12.92</td><td>44.86</td></tr><tr><td>Kimi-K2.5</td><td>69.0</td><td>73.3</td><td>70.5</td><td>17.1</td><td>57.48</td><td>36.13</td><td>64.35</td><td>90.26</td><td>10.03</td><td>37.95</td></tr><tr><td>Qwen3-VL-8B</td><td>52.7</td><td>58.4</td><td>54.5</td><td>13.0</td><td>44.66</td><td>33.31</td><td>54.78</td><td>78.84</td><td>15.90</td><td>55.52</td></tr><tr><td>InternVL3-38B</td><td>38.1</td><td>41.2</td><td>38.7</td><td>7.9</td><td>31.47</td><td>30.91</td><td>50.75</td><td>71.28</td><td>19.67</td><td>62.41</td></tr><tr><td>TikZero+ 10B</td><td>25.5</td><td>6.8</td><td>26.4</td><td>3.1</td><td>15.43</td><td>17.19</td><td>33.99</td><td>52.59</td><td>24.55</td><td>78.41</td></tr></table>

Table G.2 Three-level metric results of 11 models on D2C-E with preserve-only (p) and edit-only (e) splits. ↓ means lower is better. Bold is best in class; underline is second best.
<table><tr><td></td><td colspan="5">Preserve-Only</td><td colspan="5">Edit-Only</td><td colspan="3">CBLEU  $( p / e )$ </td><td colspan="3">Image-Level</td></tr><tr><td>Model</td><td> $\mathrm { F 1 _ { t y p e } }$ </td><td> $\mathrm { F 1 } _ { \mathrm { t e x t } }$ </td><td> $\mathrm { F 1 } _ { \mathrm { c o l o r } }$ </td><td> $\mathrm { F 1 } _ { \mathrm { b b o x } }$ </td><td> $\mathrm { F 1 } _ { \mathrm { a v g } }$ </td><td> $\mathrm { F 1 _ { t y p e } }$ </td><td> $\mathrm { F 1 } _ { \mathrm { t e x t } }$ </td><td> $\mathrm { F 1 _ { c o l o r } }$ </td><td> $\mathrm { F 1 } _ { \mathrm { b b o x } }$ </td><td> $\mathrm { F 1 _ { a v g } }$ </td><td>p</td><td>e</td><td>SSIM</td><td>CLIP</td><td>FID↓</td><td>LPIPS↓</td></tr><tr><td>Closed-Source</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini-3.1 Pro</td><td>70.4</td><td>70.7</td><td>69.1</td><td>32.2</td><td>60.60</td><td>43.6</td><td>60.4</td><td>52.1</td><td>7.3</td><td>40.84</td><td>42.02</td><td>2.98</td><td>57.63</td><td>75.32</td><td>4.62</td><td>41.80</td></tr><tr><td>Gemini-3.0 Pro</td><td>77.4</td><td>78.4</td><td>75.7</td><td>28.9</td><td>65.12</td><td>39.9</td><td>64.9</td><td>51.2</td><td>5.0</td><td>40.24</td><td>44.77</td><td>2.34</td><td>61.21</td><td>83.55</td><td>5.84</td><td>39.29</td></tr><tr><td>Gemini-3.0 Flash</td><td>71.2</td><td>72.7</td><td>69.0</td><td>24.2</td><td>59.29</td><td>37.1</td><td>58.7</td><td>48.8</td><td>4.3</td><td>37.24</td><td>42.21</td><td>2.17</td><td>57.17</td><td>77.71</td><td>5.54</td><td>43.13</td></tr><tr><td>GPT-5.2</td><td>66.8</td><td>70.3</td><td>64.5</td><td>20.8</td><td>55.59</td><td>28.7</td><td>53.3</td><td>40.3</td><td>2.3</td><td>31.15</td><td>38.75</td><td>1.15</td><td>54.71</td><td>75.94</td><td>8.61</td><td>47.77</td></tr><tr><td>Claude-4.6 Opus</td><td>75.1</td><td>78.2</td><td>72.7</td><td>24.3</td><td>62.56</td><td>25.4</td><td>60.5</td><td>39.0</td><td>3.0</td><td>31.95</td><td>43.59</td><td>1.42</td><td>59.54</td><td>84.02</td><td>8.75</td><td>46.14</td></tr><tr><td>Seed-2.0 Pro</td><td>64.9</td><td>66.2</td><td>61.3</td><td>19.8</td><td>53.03</td><td>28.7</td><td>50.0</td><td>39.5</td><td>2.3</td><td>30.14</td><td>37.49</td><td>1.41</td><td>51.48</td><td>72.24</td><td>10.15</td><td>54.21</td></tr><tr><td>Open-Source</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-397B-A17B</td><td>74.0</td><td>75.4</td><td>71.1</td><td>30.5</td><td>62.75</td><td>35.2</td><td>60.1</td><td>45.3</td><td>4.7</td><td>36.35</td><td>43.66</td><td>1.91</td><td>59.14</td><td>81.72</td><td>7.08</td><td>44.18</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>75.3</td><td>77.0</td><td>70.3</td><td>32.6</td><td>63.79</td><td>30.0</td><td>57.9</td><td>40.9</td><td>4.8</td><td>33.38</td><td>43.05</td><td>1.85</td><td>58.06</td><td>82.08</td><td>8.63</td><td>50.03</td></tr><tr><td>Kimi-K2.5</td><td>73.8</td><td>74.9</td><td>70.5</td><td>33.3</td><td>63.12</td><td>35.7</td><td>60.4</td><td>44.6</td><td>5.4</td><td>36.52</td><td>42.11</td><td>2.02</td><td>58.77</td><td>80.98</td><td>5.89</td><td>45.11</td></tr><tr><td>Qwen3-VL-8B</td><td>24.9</td><td>26.3</td><td>23.5</td><td>10.4</td><td>21.26</td><td>8.6</td><td>17.2</td><td>12.8</td><td>1.2</td><td>9.95</td><td>16.29 33.13</td><td>0.51</td><td>22.07</td><td>31.81</td><td>18.16</td><td>82.64</td></tr><tr><td>InternVL3-38B</td><td>53.6</td><td>55.3</td><td>49.1</td><td>20.9</td><td>44.72</td><td>20.5</td><td>38.5</td><td>29.4</td><td>2.3</td><td>22.69</td><td></td><td>1.29</td><td>48.37</td><td>67.54</td><td>13.39</td><td>65.13</td></tr></table>

Table G.2 reports the per-object F1 scores on D2C-E with preserve-only and edit-only splits, alongside code-level (CrystalBLEU) and image-level (SSIM, CLIP, FID, LPIPS) metrics. For preserve-only elements, models achieve moderate scores on type, text, and color, but $\mathrm { F 1 } _ { \mathrm { b b o x } }$ remains the lowest. For edit-only elements, all scores drop substantially, e.g., $\mathrm { F 1 _ { t y p e } }$ shows the largest absolute gap, indicating that models struggle to instantiate correct object types for required editing elements.

## G.3 DQA Analysis Per Diagram Type

Table G.3 reports per-domain DQA accuracy across all 11 evaluated models (TikZero+ 10B is excluded as it is not trained on question answering). Fig. G.2 provides a complementary visual comparison using grouped bar charts.

Graph structures is the easiest domain for most models, with top-performing models exceeding 90% accuracy (e.g., Gemini-3.1 Pro at 91.34%). This is likely because graph-related questions often involve counting nodes or identifying connectivity patterns, which are relatively straightforward visual reasoning tasks.

3D Shapes is consistently the hardest DQA domain, with even the best models scoring below 69%. The dificulty stems from the need to reason about three-dimensional spatial relationships, such as viewing angles and surface properties, from rendered diagrams.

Among open-source models, Qwen3-VL-8B shows particularly uneven domain performance: it achieves 66.71% on Circuits but only 27.05% on chemistry, suggesting that smaller models lack generalization across diverse diagram types. Conversely, Qwen3.5-397B-A17B maintains relatively balanced accuracy (66–88%) across all domains, approaching closed-source performance.

## G.4 Agentic Tool-Call Statistics

Table G.4 reports tool-calling statistics in the agentic settings (S3/S8/S16 in Table 4).

Table G.3 DQA accuracy (%) per diagram type. Bold is best in class; underline is second.
<table><tr><td>Model</td><td>Charts</td><td>P.Geom.</td><td>3D</td><td>Graph</td><td>Chem.</td><td>Circuits</td><td>Avg.</td></tr><tr><td>Closed-Source</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini-3.1 Pro</td><td>85.95</td><td>83.63</td><td>67.69</td><td>91.34</td><td>86.07</td><td>84.29</td><td>86.29</td></tr><tr><td>Gemini-3.0 Pro</td><td>85.08</td><td>87.16</td><td>67.69</td><td>90.88</td><td>85.79</td><td>84.58</td><td>86.46</td></tr><tr><td>Gemini-3.0 Flash</td><td>82.95</td><td>78.80</td><td>68.57</td><td>89.29</td><td>86.89</td><td>84.01</td><td>84.07</td></tr><tr><td>GPT-5.2</td><td>82.08</td><td>81.93</td><td>63.96</td><td>83.87</td><td>81.42</td><td>83.43</td><td>81.67</td></tr><tr><td>Claude-4.6 Opus</td><td>70.42</td><td>79.96</td><td>60.00</td><td>61.22</td><td>73.77</td><td>78.39</td><td>68.75</td></tr><tr><td>Seed-2.0 Pro</td><td>70.75</td><td>76.12</td><td>60.22</td><td>79.73</td><td>80.05</td><td>82.42</td><td>75.90</td></tr><tr><td>Open-Source</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-397B-A17B</td><td>82.03</td><td>83.45</td><td>66.15</td><td>87.50</td><td>86.34</td><td>81.12</td><td>83.42</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>58.55</td><td>72.99</td><td>55.82</td><td>60.88</td><td>66.12</td><td>76.08</td><td>63.60</td></tr><tr><td>Kimi-K2.5</td><td>76.20</td><td>81.75</td><td>64.40</td><td>83.73</td><td>80.60</td><td>80.12</td><td>79.74</td></tr><tr><td>Qwen3-VL-8B</td><td>33.06</td><td>66.46</td><td>39.12</td><td>47.70</td><td>27.05</td><td>66.71</td><td>47.12</td></tr><tr><td>InternVL3-38B</td><td>55.17</td><td>75.31</td><td>49.01</td><td>48.23</td><td>48.36</td><td>69.74</td><td>56.39</td></tr></table>

![](images/14d142d9215aece7fb53410b66573f5612c396871e6398dc83f2c6c4b71a2b44.jpg)  
Figure G.2 (Updated) DQA accuracy (%) per diagram type (an alternative visualization of Table G.3). Bars distinguish closed-source from open-source models.

Adoption. All six models invoke the tool to some degree, but adoption varies widely (0.2%–96.5%) across tasks and models. On D2C-P and D2C-E, Claude-4.6 Opus adopts tools most aggressively (93.7%/96.5%), followed by Gemini-3.0 Pro (45.0%/48.0%) and Qwen3-VL-8B (38.7%/43.8%), while Seed-2.0 Pro is most conservative (8.3%/10.0%). DQA triggers much lower tool use across all models, since question-answering rarely requires code-level documentation lookup. Within D2C-E, domains requiring specialized packages, e.g., circuit (circuitikz, up to 80%) and chemistry (chemfig, up to 65%), elicit the highest adoption, while graph diagrams drawn with basic tikz commands show the lowest (1–13% for most models).

Intensity. Claude-4.6 Opus averages n¯=2.83 calls per question. We inspect its calling pattern: it first queries package syntax, then specific commands, final usage examples. This structured pattern correlates with its consistent gains under tool use. Gemini-3.1 Pro shows the highest average calls among tool-invoking questions (n¯<sub>+</sub>=2.37 in D2C-E) despite low adoption (15%), indicating excessive querying loops, e.g., repeated searches without synthesizing results, leading to context rot at the token limit. Qwen3-VL-8B shows the opposite: moderate adoption (43.8%) but low n¯<sub>+</sub>=1.12, typically issuing only a single query per question.

Response and execution rates. Table G.5 compares two rates between direct (S1/S6) and tool-use agentic settings (S3/S8) for D2C-P/D2C-E: (1) Response rate (%): the percentage of questions where the model outputs a complete response; failures occur when the model exceeds the context length or enters tool-calling loops without producing final code; (2) Execution rate (%): the percentage of generated code that compiles successfully; failures occur when the output code contains syntax errors. Gemini-3.1 Pro sufers the most: tool access reduces the response rate by 25.0% (D2C-P) and 23.8% (D2C-E) due to the querying loops described above. Claude-4.6 Opus benefits most: tool access improves execution rate by +4.2% on D2C-E. Other models show minimal changes.

Table G.4 Tool-calling statistics in tool-use agentic settings (S3/S8 = +Tool; S16 = +Tool&TikZ Codes). N is the number of questions; Adopt. is the percentage of questions where the model invokes the tool at least once (≥1) out of all N questions; n¯ is the total call count divided by $N ; \bar { n } _ { + }$ is the total call count divided by the number of questions where the model called the tool at least once.
<table><tr><td rowspan="2">Model</td><td colspan="3">D2C-P (S3, N=300)</td><td colspan="3">D2C-E (S8, N=600)</td><td colspan="3">DQA (S16, N=600)</td></tr><tr><td>Adopt. (%)</td><td>n</td><td>n4</td><td>Adopt. (%)</td><td>n</td><td>n4</td><td>Adopt. (%)</td><td>n</td><td>n4</td></tr><tr><td>Claude-4.6 Opus</td><td>93.7%</td><td>2.70</td><td>2.88</td><td>96.5%</td><td>2.83</td><td>2.93</td><td>2.7%</td><td>0.05</td><td>1.75</td></tr><tr><td>Gemini-3.0 Pro</td><td>45.0%</td><td>0.75</td><td>1.67</td><td>48.0%</td><td>0.79</td><td>1.65</td><td>14.0%</td><td>0.22</td><td>1.56</td></tr><tr><td>Qwen3-VL-8B</td><td>38.7%</td><td>0.43</td><td>1.10</td><td>43.8%</td><td>0.49</td><td>1.12</td><td>0.2%</td><td>0.00</td><td>1.00</td></tr><tr><td>GPT-5.2</td><td>26.3%</td><td>0.49</td><td>1.87</td><td>23.3%</td><td>0.43</td><td>1.84</td><td>5.7%</td><td>0.08</td><td>1.47</td></tr><tr><td>Gemini-3.1 Pro</td><td>18.7%</td><td>0.44</td><td>2.34</td><td>15.0%</td><td>0.35</td><td>2.37</td><td>1.2%</td><td>0.02</td><td>1.43</td></tr><tr><td>Seed-2.0 Pro</td><td>8.3%</td><td>0.09</td><td>1.04</td><td>10.0%</td><td>0.10</td><td>1.02</td><td>0.8%</td><td>0.01</td><td>1.00</td></tr></table>

Table G.5 Response and execution rates (%) in direct (S1/S6) vs. tool-use settings (S3/S8). ∆: change from direct to +Tool (↑ gain, ↓ drop).
<table><tr><td></td><td colspan="6">D2C-P</td><td colspan="6">D2C-E</td></tr><tr><td></td><td colspan="3">Resp. Rate.</td><td colspan="3">Exec. Rate.</td><td colspan="3">Resp. Rate.</td><td colspan="3">Exec. Rate.</td></tr><tr><td>Model</td><td>S1</td><td>S3</td><td>Δ</td><td>S1</td><td>S3</td><td>Δ</td><td>S6</td><td>S8</td><td>∆</td><td>S6</td><td>S8</td><td>∆</td></tr><tr><td>Claude-4.6 Opus</td><td>100.0</td><td>100.0</td><td>0.0</td><td>91.3</td><td>93.3</td><td>+2.0</td><td>100.0</td><td>99.7</td><td>-0.3</td><td>89.2</td><td>93.3</td><td>+4.2</td></tr><tr><td>Seed-2.0 Pro</td><td>100.0</td><td>100.0</td><td>0.0</td><td>89.7</td><td>92.3</td><td>+2.7</td><td>99.7</td><td>100.0</td><td>+0.3</td><td>90.3</td><td>89.2</td><td>-1.2</td></tr><tr><td>GPT-5.2</td><td>100.0</td><td>97.7</td><td>-2.3</td><td>95.3</td><td>91.7</td><td>-3.7</td><td>100.0</td><td>98.2</td><td>-1.8</td><td>91.3</td><td>92.7</td><td>+1.3</td></tr><tr><td>Gemini-3.0 Pro</td><td>100.0</td><td>99.7</td><td>-0.3</td><td>86.3</td><td>83.7</td><td>-2.7</td><td>100.0</td><td>99.3</td><td>-0.7</td><td>90.0</td><td>86.8</td><td>-3.2</td></tr><tr><td>Gemini-3.1 Pro</td><td>100.0</td><td>75.0</td><td>-25.0</td><td>73.3</td><td>60.7</td><td>-12.7</td><td>100.0</td><td>76.2</td><td>-23.8</td><td>79.7</td><td>63.2</td><td>-16.5</td></tr><tr><td>Qwen3-VL-8B</td><td>99.7</td><td>100.0</td><td>+0.3</td><td>88.0</td><td>87.3</td><td>-0.7</td><td>99.3</td><td>100.0</td><td>+0.7</td><td>86.8</td><td>86.3</td><td>-0.5</td></tr></table>

## H Limitations and Future Work

Agentic evaluation. As acknowledged (§3.5), we do not claim Diagram-MMU evaluates the full spectrum of agentic ability in the vibe writing framework. Vibe writing platforms (e.g., Prism [12]) are becoming increasingly popular, as any researcher can prompt models in natural language to generate academic papers directly in LAT<sub>E</sub>X format. Common needs include inserting a diagram into the manuscript as compilable LAT<sub>E</sub>X TikZ code, applying specific edits based on the paper content, or describing the domain-specific meaning of diagram symbols. Our evaluation settings target such multimodal aspects of vibe writing through diagram-to-code and diagram understanding tasks.

However, the vibe writing workspace involves broader capabilities such as text drafting, citation management, and layout formatting, which remain outside our scope and would require designing agentic evaluation settings for purely textual abilities. Moreover, broader agentic capabilities such as long-horizon task decomposition, multi-tool orchestration across heterogeneous environments, self-correction via execution feedback, and interactive collaboration with human users are also not covered by the current benchmark.

Code representation. Diagram-MMU adopts LAT<sub>E</sub>X TikZ as the code representation, whose output compiles inline within LAT X authoring environments. But, in practice, researchers also create scientific diagrams with Python libraries (e.g., Matplotlib, Seaborn), SVG/HTML tools, or other LAT<sub>E</sub>X drawing packages (e.g., pstricks, xy-pic), which are currently not covered by the evaluation settings of Diagram-MMU.

Future directions. In future work, we plan to extend agentic settings to evaluate: (1) broader textual and layout capabilities in scientific writing, such as text-to-TikZ generation, table construction from data, and mathematical derivation formatting; and (2) richer agentic workflows, including execution feedback loops, iterative human-in-the-loop refinement, and multi-tool orchestration, that reflect the real-world demands of vibe writing. For code representation, cross-language conversion pipelines that translate Python or SVG diagrams into TikZ would expand coverage while preserving inline LAT<sub>E</sub>X compilation, and supporting diagram import via image upload would allow vibe writing workspaces to ingest figures. Alternatively, extending evaluation to additional LAT<sub>E</sub>X drawing packages beyond TikZ (e.g., pstricks, xy-pic) would broaden coverage within the LAT<sub>E</sub>X ecosystem directly. For diagram understanding and reasoning, we plan to move toward expert-level questions, and to incorporate richer context such as figure captions from the surrounding paper, enabling evaluation of diagram comprehension grounded in the full manuscript.

We hope future versions of Diagram-MMU can provide more comprehensive insights and guide targeted improvements by pinpointing where models fail: whether in foundational abilities (perception, coding, knowledge, reasoning) or agentic abilities (context utilization, tool use, state management, planning). We do not foresee negative societal consequences; the benchmark is constructed from publicly available TikZ documentation and contains no private or sensitive data.

## I Acknowledgement

We thank all 13 annotators for their contributions to data quality assurance; together, they verified over 2,000 scientific diagrams through cross-validation across six domains. Three annotators are co-authors of this paper. Among the remaining ten contributors, one is a research scientist at NetMind.ai, eight are PhD students at Nanjing University of Science and Technology (NJUST), and one is a PhD student at Westlake University. We gratefully acknowledge these ten collaborators and detail below the number of annotated samples and the domains each covered:

• Xinyu Miao from NJUST — 419 cases, covering 3D Shape, Charts, Chemical Expressions, Graph Structures, Planar Geometry

• Yiyou Gao from NJUST — 331 cases, covering 3D Shape, Charts, Graph Structures, Planar Geometry

• Ruolin Wang from NJUST — 271 cases, covering 3D Shape, Charts, Graph Structures, Planar Geometry

• Xinyao Hu from NetMind.ai — 208 cases, covering 3D Shape, Charts, Graph Structures, Planar Geometry

• Jinhong Yang from NJUST — 200 cases, covering Graph Structures

• Zongxin Zhu rom NJUST — 194 cases, covering Charts, Graph Structures, Planar Geometry

• Yu Wang from NJUST — 181 cases, covering Graph Structures, Planar Geometry

• Xinyu Zhang from NJUST — 170 cases, covering Charts, Circuit Diagrams, Graph Structures

• Xueji Fang from Westlake University — 107 cases, covering 3D Shape, Charts, Graph Structures, Planar Geometry

• Wei Tang from NJUST — 11 cases, covering 3D Shape, Charts