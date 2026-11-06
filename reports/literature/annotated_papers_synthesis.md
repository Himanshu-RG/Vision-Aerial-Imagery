# Synthesis of the annotated papers

How five open-vocabulary detection (OVD) papers relate to each other and to our project.

**Our project.** *Adaptive Vocabulary Selection for Robust OVD in Aerial Imagery* (team Transmontane).
- Detector: frozen Grounding DINO-T.
- Benchmark: LAE-80C.
- Selector: RemoteCLIP picks a per-image class list with an adaptive threshold.

**Where the details are.**
- Per-paper questions and answers: [annotated_papers_QA.md](annotated_papers_QA.md).
- Highlighted PDFs: `papers/Annotated/highlighted/`.

---

## 1. The five papers at a glance

| | WeDetect | LBP | GenerateU | Object-centric OVD | CondHead |
|---|---|---|---|---|---|
| Authors, venue | Fu et al., CVPR 2026 | Li et al., CVPR 2024 | Lin et al., CVPR 2024 | Rasheed et al., NeurIPS 2022 | Wang, CVPR 2023 |
| Main idea | Detection as retrieval: a dual-tower detector with no fusion layers | Learn prompts for the unnamed objects hidden in the "background" | Generate object names instead of matching against a given list | Make CLIP knowledge object-centric instead of image-centric | Condition the box and mask heads on the class text embedding |
| Detector | ConvNeXt + YOLO-style head; Qwen3-VL for REC | Faster R-CNN (BARON-based) | Class-agnostic Deformable DETR + FlanT5 | Faster R-CNN / Mask R-CNN / CenterNetV2 | Plug-in head for OVR-CNN, ViLD, RegionCLIP |
| Where the vocabulary enters | Text embeddings dotted with grid features (cached) | Softmax classifier over base + novel + learned background prompts | Nowhere at inference; names are generated | CLIP text embeddings replace the classifier weights | Classifier *and* box/mask head parameters |
| Training needed? | Yes: 15M images, 330M boxes | Yes | Yes: VG + 5M pairs, 16 A100 | Yes: 8 A100, 3× schedule | Yes, on top of each baseline |
| Inference-time text intervention? | No | Probability rectification (IPR) | No list at all | No | Description tuning (an observation) |
| Main benchmarks | LVIS, COCO, COCO-O, ODinW, RefCOCO/+/g, HumanRef | OV-COCO (48/17), OV-LVIS (866/337) | Zero-shot LVIS minival, COCO, Objects365 | OV-COCO, OV-LVIS, cross-dataset to OpenImages and Objects365 | COCO, LVIS, cross-dataset to VOC, COCO and Objects365 |
| Headline number | LVIS 49.4 AP (Large); Tiny at 62.5 FPS | Novel AP50 37.8; APr 24.1 | 26.8 AP zero-shot LVIS (Swin-T) | Novel AP50 36.6 (+8.2) | +3.0 APr for ~1% FLOPs |
| Aerial evaluation? | No (ODinW has a few drone sets, not reported separately) | No | No | No | No |

---

## 2. Themes across the papers

### 2.1 The vocabulary is part of the model, not just an input list

All five papers agree that the class list strongly shapes what the detector outputs. They differ in *where* it enters:

- **As classifier weights.** Rasheed et al. say a typical OVD solution is to "replace the classifier weights with text embeddings of the target vocabulary". LBP does the same and adds learned background entries. Here, choosing the vocabulary means choosing the rows of the classifier.
- **Inside the features (fusion).** WeDetect explains that in fusion models like Grounding DINO, text "makes visual features query-specific". The vocabulary therefore changes the visual features themselves.
- **Inside the localisation heads.** CondHead feeds the class embedding into box regression and mask prediction. A wrong class text "misguides" the box onto the wrong object (sunflower → carrot).
- **Not at all.** GenerateU writes names for every box. The vocabulary only comes back when generated names are mapped to a label set for evaluation.

**For us.** Grounding DINO is the strongest case of text dependence: the text enters fusion, query selection and cross-attention. Choosing the class names per image is therefore a meaningful intervention, not just cosmetic filtering.

### 2.2 A large vocabulary costs time

- **Fusion re-runs per chunk.** WeDetect: Grounding DINO on LVIS's 1,203 classes needs 31 chunked forward passes ("several seconds" per image), because fusion features cannot be shared across prompts.
- **Dual-tower models avoid it.** WeDetect-Tiny runs at 62.5 FPS against Grounding DINO-T's 6.0 FPS, because text embeddings are computed once.
- **Overheads are reported as gain vs cost.** CondHead (+1.1% FLOPs) and the WeDetect family both report their extra cost next to their gain.

**For us.** This explains why our smaller vocabulary was also faster: 414 → 343 ms per image, including the 7.9 ms selector. Reporting the selector's time inside the end-to-end figure follows the same practice.

### 2.3 Absent and overlapping class names hurt

Four papers show, in different ways, that putting the wrong names in front of a detector hurts it:

| Paper | Evidence |
|---|---|
| WeDetect | LMMs "tend to predict objects for every query, failing to reject negative queries"; Qwen2.5-VL-7B scores 17.7 AP on COCO. Training without negative queries costs 5.4 AP and 7.0 APs. |
| LBP | Learned background concepts that overlap a novel class "hinder the accurate computation of probability scores", because that class is counted twice in the softmax. Background prompts alone lower novel AP50 (35.8 → 34.3) until IPR corrects for the overlap (+2.5). |
| CondHead | A wrong category text worsens box regression. Tuned descriptions improve detection and segmentation. |
| GenerateU | Hand-made lists bring "language ambiguities" ("person" / "a man" / "young boy") and can be incomplete. |

**For us.** These findings match our failure analysis on LAE-80C:
- With all 80 classes, 77% of confident false positives were classes never annotated in that image's dataset. Selection cut confident false positives by 76%.
- 59% of the classes we selected belong to *another* source's label set (e.g. "Ferry" on an image labelled "Passenger Ship"). This is the overlap problem LBP formalises and GenerateU describes.

Those papers fix it with training: negative queries, background prompts, conditional heads. We remove the names at inference, which is the option available for a frozen detector.

### 2.4 Image-level evidence is weak for objects, especially small ones

This is the most consistent message across the five papers:

- **Rasheed et al.** CLIP "lacks precise localization" because it is trained on full, scene-centric images. Image-level labels and captions "do not convey the precise object-centric information".
- **WeDetect.** In object retrieval ("does this image contain class X?"), whole-image CLIP gets F1 46.4 on COCO, while region-level WeDetect-Uni gets 83.6.
- **GenerateU.** A 224×224 image encoder "is insufficient for the object detection task", and caption-supervised models lack "fine-grained object categories".
- **GenerateU.** Captions mention far fewer nouns than there are objects in the image.

**For us.** These points explain several of our results:

| Our observation | Explanation from these papers |
|---|---|
| The whole-image RemoteCLIP selector keeps the right class for only 26% of ground-truth objects; dense FAIR1M/xView scenes hold most objects | Image-level evidence misses small and dense objects |
| Local crops raised small-class selection recall (0.22 → 0.31) | Region-level evidence helps presence decisions |
| The VocAda-style caption selector was slow (~1.2 s) and lost on AP | Captions under-mention objects |

They also point to the next step: score classes on object proposals, not on the whole image or fixed crops.

### 2.5 Object proposals are a shared tool

Every paper relies on some class-agnostic "where are the objects" step:

| Paper | Proposal step |
|---|---|
| Rasheed et al. | MViT queried with "all objects", top-5 proposals |
| WeDetect-Uni | A single learned objectness prompt on a frozen detector |
| GenerateU | A class-agnostic Deformable DETR |
| LBP | A class-agnostic RPN, then k-means clustering of the background proposals |

Proposal quality matters a lot. Rasheed et al. get 4.0 novel AP50 with RPN proposals and 16.7 with MViT proposals in the same distillation setup.

**For us.** A cheap objectness pre-pass, followed by RemoteCLIP scoring of proposals, is a direct way to combine §2.4 with our training-free setting.

### 2.6 Names, synonyms and granularity

- **WeDetect** labels each object at several levels ("animal, dog, a yellow dog"). Removing either level costs 1–2 AP on LVIS.
- **GenerateU's** beam search "naturally generates synonyms". Beam size 3 raises APr from 12.3 to 20.0.
- **GenerateU's** evaluation has to map free-form names to dataset classes. A "laptop" might be labelled "computer" or "PC".

**For us.** LAE-80C merges five datasets, each with its own label space, so one object type has several names: "Small Car" (FAIR1M), "Passenger Car" (xView), "vehicle" (DIOR). This is why we report source-aware AP next to standard AP. A taxonomy-aware or synonym-merged vocabulary is the natural follow-up. Our own synonym-expansion experiment did not pass the replacement rule we had fixed in advance.

### 2.7 Evaluation practice

| Practice | Where it appears | What we do |
|---|---|---|
| OV-COCO 48 base / 17 novel, AP50 | LBP, Rasheed, CondHead | AP50@900 on LAE-80C |
| OV-LVIS 866 base / 337 rare, APr | LBP, Rasheed, CondHead | Per-class AP plus a per-class sign test |
| Cross-dataset transfer by swapping the vocabulary | Rasheed, CondHead, GenerateU | NWPU VHR-10 with no retuning |
| Thresholds fixed on one set and reused | WeDetect (COCO threshold reused on LVIS) | F1 threshold fixed on dev, reused on test |
| Hyper-parameters chosen on held-out data | Rasheed (random search) | Everything tuned on dev; test used once |
| No overlap between pre-training and evaluation data | Rasheed (MViT retrained on filtered LMDet) | We disclose RemoteCLIP's DIOR/DOTA exposure and NWPU's presence in LAE-1M |
| The evaluator can move the result | GenerateU: 26.8 (CLIP) vs 21.7 (BERT) AP for the same predictions | We report both standard and source-aware AP, fixed before the test run |

**Missing from all five.** None of the papers reports confidence intervals or significance tests. Each reports a single number per setting; only Rasheed et al. average over 3 runs for LVIS. Our paired bootstrap with three robustness checks goes beyond this common practice.

### 2.8 Training-heavy vs training-free

Every method here needs training:
- **Heavy:** WeDetect (15M images), GenerateU (5M pairs, 16 A100), Rasheed (8 A100).
- **Lighter:** LBP and CondHead, which still retrain their baselines.

The only test-time levers mentioned are LBP's probability rectification and CondHead's observation that tuned descriptions help. Neither one chooses which classes to query.

---

## 3. What this means for our project

### Findings that support our approach
1. **The class names change detections, including localisation.** In text-conditioned detectors, choosing the vocabulary is a lever, not a cosmetic filter (CondHead, WeDetect, LBP).
2. **Absent and overlapping names cause false positives and score splitting.** We see the same thing in LAE-80C (WeDetect, LBP, GenerateU).
3. **Fusion detectors pay for every prompt token.** A shorter vocabulary is also a faster one (WeDetect).
4. **The gain depends on how well the encoder knows the domain** (CondHead). This matches our split result: source-aware AP50 rises from 0.035 to 0.120 on the DIOR/DOTA group and only from 0.008 to 0.009 on FAIR1M/xView.

### Findings that limit our approach
1. **Whole-image CLIP scoring is a weak presence detector** for small objects (WeDetect object retrieval, Rasheed, GenerateU). This is a ceiling on our current selector.
2. **The papers fix these problems with training**, which may be better when training is allowed. Our claim is limited to the frozen-detector, no-training setting.

### Gap confirmed

None of the five papers selects a per-image vocabulary at inference for a frozen detector. None of them evaluates on aerial or remote-sensing benchmarks. This is consistent with the gap in `reports/M0_gap_hypotheses_novelty.md`.

### Ideas for future work taken from these papers

| Idea | Source | Fits a frozen detector? |
|---|---|---|
| Score classes on object proposals (a generic objectness pass) instead of the whole image or fixed crops | WeDetect-Uni, Rasheed (MViT "all objects") | Yes, if the proposal model is also frozen |
| Merge or down-weight near-duplicate names using text-embedding similarity before prompting | LBP (IPR) | Yes |
| Hierarchical or multi-granularity names for merged aerial label spaces | WeDetect labels, GenerateU synonyms | Yes, at inference |
| Tune the class descriptions at inference | CondHead (Fig. 4) | Yes |
| Train with negative queries, or use region-level distillation | WeDetect, Rasheed, LBP | No: needs training |

---

## 4. Quick reference of numbers used above

| Paper | Number | Where in the paper |
|---|---|---|
| WeDetect | Grounding DINO on LVIS: 31 forward passes at chunk size 40 | Sec. 1 |
| WeDetect | Grounding DINO-T 6.0 FPS; WeDetect-Tiny 62.5 FPS; Large 49.4 LVIS AP | Table 1 |
| WeDetect | Object retrieval F1 (COCO): CLIP 46.4, Large-Uni 83.6 | Table 4 |
| WeDetect | No negative detection data: −5.4 AP, −7.0 APs | Table 7 |
| LBP | Novel AP50: BCP alone 34.3, BCP + IPR 36.8, full 37.8 (baseline 35.8) | Table 3 |
| LBP | OV-LVIS APr 24.1 (LBP‡) vs 23.2 (BARON‡) | Table 2 |
| GenerateU | Zero-shot LVIS: 26.8 AP (Swin-T, VG+GRIT) vs GLIP-T 26.0 | Table 1 |
| GenerateU | Frozen KOSMOS-2 14.3 → end-to-end 26.8 AP | Table 2 |
| GenerateU | Same predictions: CLIP 26.8 vs BERT 21.7 AP | Table 4 |
| Rasheed | COCO novel AP50 36.6 (supervised base 1.7) | Tables 2, 3 |
| Rasheed | RPN 4.0 vs MViT 16.7 novel AP50 (L1 distillation) | Table 7 |
| Rasheed | Pseudo-box 30.4 vs max-size RPN 25.9 novel AP50 | Table 8 |
| CondHead | COCO novel gains: +1.4 / +2.1 / +2.4 (OVR-CNN / ViLD / RegionCLIP) | Table 2 |
| CondHead | LVIS APr 22.1 → 25.1 (RegionCLIP RN50x4); 193.5 → 195.8 GFLOPs | Tables 3, 7 |
