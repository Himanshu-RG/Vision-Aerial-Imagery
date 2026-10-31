# Adaptive Vocabulary Selection for Robust Open-Vocabulary Object Detection in Aerial Imagery

**Project overview for review** · Team *Transmontane*: Amit Adhikari (23BCE2327), Himanshu Ray (23BCE2335), Saurav Kumar Gupta (23BCE2336), Avishek Chowdhury (23BCE2344)
**Status as of 25 Oct 2026:**
- Modules M0–M3 complete.
- Experiment E0 complete.
- M4 (simple baselines) complete except the caption-based baseline.
- M5 (global + local selection) complete. It improves *selection* but not *detection*: a documented negative result.
- M6–M7 (uncertainty-aware retention, adaptive budget) complete. **An adaptive per-image budget is the best method so far.**
- M8–M9 complete (experiment registry, precision/recall/F1, tiling diagnostic: small objects are a detector limitation).
- Next: remaining baselines, failure analysis, generalization, one-time test evaluation.

Repository: <https://github.com/Himanshu-RG/Vision-Aerial-Imagery>. The living project log is [`CONTEXT.md`](../CONTEXT.md); module reports are in [`reports/`](.).

---

## 1. Summary
Open-vocabulary detectors such as **Grounding DINO** find objects named in a text prompt. On aerial imagery, though, a prompt that lists every candidate class makes the detector confuse similar classes. We study whether **choosing a small, image-specific vocabulary before detection** helps a **frozen, pretrained** Grounding DINO on aerial images, and at what computational cost.

What we have established so far, all on the tuning (dev) split, with no test data used:
1. **There is large headroom.** If the detector is given only the classes actually present in each image (an *oracle*, not a method), AP50 rises from **0.015 to 0.140** (×9.6), and recall of ground-truth objects from **12% to 47%**.
2. **Content matters, not size.** A random vocabulary of the same small size is *no better* than all 80 classes. A useful selector must keep the classes that are present.
3. **A benchmark artifact we identified explains only ~11% of that gap.** The LAE-80C benchmark merges five datasets that each annotate a *disjoint* set of classes. We show this distorts standard evaluation, and we propose a source-aware evaluation.
4. **A simple deployable selector already helps.** Ranking the 80 classes by whole-image CLIP / RemoteCLIP similarity and keeping the top 8–20 improves source-aware AP50 from **0.028 to 0.046–0.058** (×1.6–2.1). The gain is robust under three statistical checks and recovers up to ~27% of the oracle headroom. It also lowers end-to-end latency by ~19%, since the selector costs only ~8 ms on the GPU.
5. **The remaining gap points to our core idea.** Global CLIP selection keeps few of the present classes at small budgets and **does not help small objects at all**: small-object recall drops at k = 3, and the FAIR source, with 91% small objects, does not improve. This motivates combining **global (whole-image) and local (crop-level) evidence**.
6. **Local evidence improves selection but not detection.** Adding 13 overlapping crops to RemoteCLIP raises recall of small-object classes from 0.22 to 0.31 at 8 classes (significant; natural-image CLIP gains nothing). In detection, Grounding DINO then finds more small objects (recall 0.10 → 0.12–0.13), but localises them poorly, so AP does not improve, and the crops cost 83 ms per image. **The small-object bottleneck is the detector itself, not the vocabulary.**
7. **The best method so far: an adaptive per-image vocabulary size.** Keep every class whose RemoteCLIP probability exceeds a threshold calibrated on other images (about 8 classes on average, more for ambiguous images, fewer for clear ones). AP50 rises to **0.055 under standard COCO scoring (×3.8 vs all 80 classes) and 0.071 source-aware (×2.5)**, robust on all three statistical checks under *both* protocols, which no earlier selector achieved. Keeping uncertain classes via crop evidence did not help detection.

---

## 2. Problem and motivation
- Aerial images contain **small, dense, visually similar objects**: 59% of the boxes in our benchmark are smaller than 32×32 pixels.
- Grounding DINO receives the class list as **one text prompt** (at most 256 tokens). With 80 overlapping aerial class names (`vehicle`, `small car`, `passenger car`, `van`, …), the detector spreads its confidence across confusable names.
- A recent study (Tsourveloudis, arXiv 2601.22164, Jan 2026) showed on the same benchmark that prompting with only the ground-truth classes gives a **15× higher F1**. It used ground truth, so it is not a usable method, and the study calls for "intelligent vocabulary filtering" as future work.
- **VocAda** (ICIP 2025) does test-time vocabulary pruning, but on natural images, with other detectors, using a 7B-parameter captioner (5–11 s per image, 17–31 GB GPU).

**Research question.** Can an **image-conditioned, deployable vocabulary selector** recover the oracle's headroom for a frozen Grounding DINO on aerial imagery, and do so faster than running the full vocabulary?

We **do not** claim vocabulary pruning itself is new (VocAda). The intended contribution is aerial-specific: global–local evidence, uncertainty-aware retention of plausible classes, an adaptive vocabulary budget, and a rigorous evaluation that accounts for the benchmark's label structure.

---

## 3. Related work and positioning (verified from primary sources)
| Work | Venue | What it does | Training? | Selects vocabulary per image? | Aerial? | Relation to us |
|---|---|---|---|---|---|---|
| Grounding DINO (Liu et al.) | ECCV 2024 | Language-guided open-set detector (DINO + BERT, deep vision–language fusion) | pretrained | no (user prompt) | no | **our frozen detector** |
| VocAda (Liu et al.) | ICIP 2025 | Caption → nouns → CLIP/LLM picks classes | no | **yes** (global caption only) | no (COCO, Objects365; Detic/VLDet/CoDet) | closest *method*; our baseline E ("VocAda-style") |
| Tsourveloudis | arXiv 2601.22164 (2026) | Zero-shot evaluation of 5 OV detectors on LAE-80C; oracle vocabulary gives 15× F1 | no | oracle only | **yes** | closest *study*; we replicate its direction with COCO metrics |
| LAE-DINO / LAE-1M (Pan et al.) | AAAI 2025 | RS open-vocabulary detector; "dynamic vocabulary construction" **during training** | yes | no (training-time) | yes | source of our benchmark (LAE-80C) |
| CastDet (Li et al.) | ECCV 2024 | Student–teacher OVD with RemoteCLIP teacher | yes | no | yes | supports RemoteCLIP for aerial semantics |
| OpenRSD (Huang et al.) | ICCV 2025 | Text/image-prompted RS detector, multi-stage training (470k images) | yes | no | yes | training-based alternative |
| Scene-adaptive & region-aware prompts (Zhao et al.) | CVPR 2024 | Learned scene/region prompts modify class embeddings | yes | no (modifies embeddings) | no | scene adaptation via training |
| TagCLIP (Lin et al.) | AAAI 2024 | Training-free *local-to-global* CLIP multi-label tagging | no | tags only (no detector) | no | **overlaps our global–local idea** and will be a baseline |

**Gap (citation sweep, 2026-10-20):** no work performs **deployable, inference-time vocabulary selection for a frozen open-vocabulary detector on aerial imagery**. **Novelty risk (honest):**
- Global CLIP selection: high. It is a baseline only.
- Global–local evidence: medium-high, because of TagCLIP.
- Uncertainty-aware retention: medium.
- Adaptive budget alone: high.

The defensible contribution is the aerial-specific selector *plus* a rigorous, artifact-aware evaluation.

---

## 4. System architecture

### 4.1 Pipeline
```mermaid
flowchart LR
    I[Aerial image] --> S{Vocabulary selector<br/>pixels only}
    V[(Master vocabulary<br/>80 class names)] --> S
    S -->|top-k / adaptive set| P[Prompt builder<br/>small car . van . ship .]
    I --> D[Grounding DINO-T<br/>frozen]
    P --> D
    D --> O[Boxes + class scores<br/>top-900 per image]
    O --> E[Evaluation<br/>standard + source-aware COCO]
```
- **Selector variants (planned, in order):**
  - global CLIP / RemoteCLIP (M4);
  - **global + local crop evidence** (M5);
  - **uncertainty-aware retention** (M6);
  - **adaptive budget** (M7).
- **Hard rule:** the selector may only see image pixels, never the file name, source dataset, original image size, or annotations. We found image size alone reveals the source dataset, so this rule prevents a shortcut. A unit test enforces it.

### 4.2 Detector: Grounding DINO (Swin-T), frozen
```mermaid
flowchart TB
    IMG[Image 800-1333 px] --> SW[Swin-T backbone<br/>27.6 M params]
    TXT[Prompt ≤ 256 tokens] --> BERT[BERT-base text encoder<br/>108.9 M params]
    SW --> FE[Feature enhancer<br/>6 layers, image↔text cross-attention<br/>21.9 M]
    BERT --> FE
    FE --> QS[Language-guided query selection<br/>900 queries]
    QS --> DEC[Cross-modality decoder<br/>6 layers, 11.1 M]
    FE --> DEC
    DEC --> HEAD[Box head + query·token similarity<br/>→ per-token logits]
    HEAD --> CLS[Class score = mean sigmoid over<br/>the class name's tokens]
```
| Property | Value (verified from the checkpoint/config) |
|---|---|
| Checkpoint | HF `IDEA-Research/grounding-dino-tiny`, commit `a2bb814d…`, sha256-locked |
| Parameters | **172.3 M** in total: BERT 108.9 M, Swin-T 27.6 M (depths 2-2-6-2), feature enhancer 21.9 M, decoder 11.1 M, heads/projections 2.9 M |
| Hidden size / queries / text limit | 256 / **900** / **256 tokens** |
| Feature levels / deformable points | 4 / 4 |
| Pretraining (Swin-T) | Objects365, GoldG, Cap4M (natural images; no aerial benchmark) |
| Our inference protocol | Official COCO-evaluation protocol of the Grounding DINO repo: caption `"a . b . c ."`, positive-map **mean** token scoring, resize to 800 short side / 1333 max, fp32. We keep the **top-900** (box, class) pairs because some images have up to 668 objects |

### 4.3 Selector encoders (Module 4)
| Model | Architecture | Parameters | Pretraining | Use |
|---|---|---|---|---|
| OpenAI CLIP ViT-B/32 | ViT-B/32 image encoder (12 layers, width 768, 224 px) + 12-layer text transformer; 512-d joint space | **151.3 M** (image 87.8 M, text 63.4 M) | 400 M web image–text pairs | natural-image baseline |
| RemoteCLIP ViT-B/32 | same architecture, remote-sensing fine-tuned | 151.3 M | remote-sensing image–caption data, which per its paper includes captions generated from detection datasets such as DOTA/DIOR (training labels; LAE-80C uses their *validation* images; exact overlap to be re-verified) | aerial-domain selector |

Class relevance = cosine similarity between the image embedding and the averaged embedding of three fixed prompt templates per class. These are similarities, not calibrated probabilities.

---

## 5. Data and evaluation protocol
**Benchmark: LAE-80C** (validation benchmark of LAE-1M, MIT license, pinned commit `60ce78a`): **3,592 images, 86,558 boxes, 80 classes**, horizontal boxes.

| Source | Images | Classes annotated | Boxes / image | Small boxes |
|---|---|---|---|---|
| FAIR1M (FAIR-18C) | 2,130 | 18 (vehicles, ships, …) | 18.0 | 91% |
| xView (Xview-34C) | 758 | 34 | 59.9 | 34% |
| DIOR (DIOR-18C) | 536 | 18 | 4.0 | 30% |
| DOTA (DOTA-6C) | 137 | 6 | 3.8 | 17% |
| Power plants (PP-4C) | 31 | 4 | 4.8 | 3% |

**Our main dataset finding:** the five sources annotate **disjoint** class sets. A car is "Small Car" in FAIR, "Passenger Car" in xView and "vehicle" in DIOR, and each image is labelled only with its own source's names. Under standard COCO evaluation, a correct "vehicle" box on a FAIR image counts as a false positive.

| Protocol element | Choice (frozen before any detector run) |
|---|---|
| Split | Our own **dev 730 / test 2,862** (iterative multi-label stratification, seed 2026). All tuning happens on dev; test is untouched until the final evaluation |
| Cleaning | 972 edge-cut boxes clipped to the image; 19 exact duplicates removed; 23 tiny boxes flagged. Every change is logged |
| Class names | Lower-case, glued names split (`groundtrackfield` → "ground track field"), no synonyms. The full prompt is 249/256 tokens |
| Evaluation | **Source-aware COCO** (primary: per image, only classes annotated by its source are scored) **and standard COCO** (for comparability) |
| Metrics | AP50 and AP at maxDets 900 (headline) and 100; AR; per-size and per-class AP; vocabulary recall; latency; peak GPU memory |
| Statistics | Paired image-level bootstrap (1,000 resamples) for every comparison |

---

## 6. What has been accomplished
| Module | Content | Status |
|---|---|---|
| M0 | Literature audit (13 works, primary sources), gap, hypotheses H0–H7, novelty-risk assessment | ✅ |
| M1 | Two-machine setup (dev laptop + RTX 5060 8 GB), exact version pins, checksummed model weights, environment audits | ✅ |
| M2 | Dataset validation, manifest, frozen split, naming policy, evaluation protocol | ✅ |
| M3 | Grounding DINO pipeline, two evaluators, bootstrap statistics; **known-answer tests** (feeding ground truth back gives AP = 1.0, etc.). 40 automated tests at M3; 46 now | ✅ |
| E0 | Headroom experiment on 200 dev images (RTX 5060) | ✅ |
| M4 | Baselines A–D (full, frequency prior, random, CLIP/RemoteCLIP top-k): selection quality on 730 dev images and detection on 200; robustness statistics. Caption-based baseline E deferred | ✅ (E pending) |
| M5 | Global + local (crop) selection: 25 variants per encoder, chosen by a pre-declared rule; selection-level gain, no detection-level gain (negative result, reported) | ✅ |

**Reproducibility:**
- Every run is identified by a hash of its configuration and records its commit, machine, GPU state and checkpoint hashes.
- Every table is generated from saved results by a script, never typed by hand.
- Failed images are counted, never dropped.

---

## 7. Results so far (dev split only)

### 7.1 E0 — how much can vocabulary selection help? (200 dev images, RTX 5060)
| Vocabulary given to the detector | Classes | AP50 (standard) | AP50 (source-aware) | Recall of GT objects (AR@900) | Latency / image |
|---|---|---|---|---|---|
| Full vocabulary | 80 | 0.015 | 0.028 | 0.124 | 418 ms |
| *Oracle:* the image's source class list | ~21 | 0.031 | 0.031 | 0.293 | 344 ms |
| *Oracle:* classes present in the image | ~2 | **0.140** | **0.140** | **0.467** | 333 ms |
| *Control:* random classes, same size | ~2 | 0.007 | 0.008 | 0.015 | 332 ms |

Paired bootstrap, oracle minus full: **+0.126** [0.119, 0.187] standard; **+0.112** [0.102, 0.166] source-aware.

**Interpretation:**
- The headroom is large and statistically clear.
- About 89% of it remains under source-aware evaluation, so it is genuinely about per-image class relevance and not a benchmark artifact.
- Giving the detector the right *dataset's* class list barely helps; selection must be per image.
- Small vocabularies are about **20% faster**, so a selector must cost under ~85 ms per image for a net speed-up.
- **Ceiling:** even with the oracle, AP50 is only 0.14, and 18 of 75 classes stay near zero (fine-grained vehicles, "working/unworking chimney", …). No vocabulary selector can fix those.

### 7.2 M4 step 1 — do simple selectors keep the classes present? (730 dev images, no detector)
![vocabulary recall vs budget](../outputs/figures/m4_vocab_recall_dev.png)

| Selector | keeps present classes at k=3 | k=8 | k=20 |
|---|---|---|---|
| Fixed frequency prior (fit on dev, so optimistic) | 0.288 | 0.610 | 0.770 |
| OpenAI CLIP ViT-B/32 (global) | 0.189 | 0.356 | 0.609 |
| RemoteCLIP ViT-B/32 (global) | 0.221 | 0.410 | 0.668 |
| Random (3 seeds) | 0.045 | 0.113 | 0.264 |

**Interpretation:**
- **RemoteCLIP beats CLIP at every budget below 80** (paired bootstrap CIs exclude 0 for k = 1, 2, 3, 5, 8).
- But both are far from the oracle and below a fixed frequency prior.
- At k = 5, global CLIP keeps only **4% (CLIP) / 16% (RemoteCLIP)** of the classes whose objects are small, vs 35–44% for others. **A single whole-image embedding does not see small objects.** That is the motivation for local, crop-level evidence (M5).

### 7.3 M4 step 2 — detection with selected vocabularies (same 200 dev images, RTX 5060)
Primary protocol (source-aware), AP50@900. A difference is called **robust** only if (i) the percentile bootstrap CI, (ii) the basic bootstrap CI and (iii) a per-class sign test all agree.

| Vocabulary | Classes | AP50 | vs full | Robust? |
|---|---|---|---|---|
| Full vocabulary | 80 | 0.028 | – | – |
| **RemoteCLIP top-8** | 8 | **0.058** | +0.030 | **yes** (35 classes better / 16 worse) |
| CLIP top-20 | 20 | 0.049 | +0.021 | **yes** (39 / 11) |
| CLIP top-8 | 8 | 0.046 | +0.018 | **yes** (31 / 13) |
| CLIP top-3 | 3 | 0.069 | +0.040 | no: the gain sits in a few large-object classes |
| Frequency prior, top-3 to 40 | 3–40 | 0.004–0.017 | worse | worse (robust under standard scoring at k = 3/8) |
| Random, top-3 to 20 | 3–20 | 0.002–0.011 | worse | – |
| *Oracle: classes present* | 2.2 | 0.140 | +0.112 | yes |

**What this shows:**
- **H2 (a deployable selector beats the full vocabulary) holds, moderately, under the primary protocol.** It is not robust under standard COCO scoring.
- The best robust budget is *moderate* (8–20 classes), not minimal. That matches our hypothesis of a trade-off between keeping plausible classes and removing confusers, which motivates uncertainty-aware retention and an adaptive budget.
- Selection **does not help small objects**: the gains come from medium and large objects (DIOR, DOTA).
- **Methodological lesson:** image-level "vocabulary recall" is a misleading proxy. The frequency prior scores highest on it yet detects worst, because class-averaged AP rewards covering many classes across images.
- **Statistics note:** the pre-planned percentile bootstrap turned out to be biased for class-averaged AP, so two robustness checks were added. They were added *after* seeing these results, but they are stricter than the pre-planned rule, so the conclusion holds under both. Two minor E0 side-claims did not survive them and were corrected; the main E0 result did.

### 7.4 M5 step 1 — does local (crop) evidence recover small-object classes? (730 dev images, no detector)
The whole image plus overlapping crops (2×2 and 3×3 grids) are scored. Each view's class probabilities are combined as α·global + (1−α)·(mean over crops). The variant was chosen by a rule fixed beforehand (best class-wise ranking mAP on dev).

![global vs local vs global+local](../outputs/figures/m5_recall_dev.png)

| Encoder | Selector | Keeps present classes at k=3 | k=8 | Small-object classes kept at k=8 |
|---|---|---|---|---|
| RemoteCLIP | global (M4) | 0.221 | 0.410 | 0.222 |
| RemoteCLIP | **global + local** (α = 0.25, 13 crops) | **0.287** | **0.455** | **0.305** |
| CLIP | global (M4) | 0.189 | 0.356 | 0.099 |
| CLIP | global + local | 0.186 | 0.343 | 0.094 |

**What this shows:**
- **For RemoteCLIP, every improvement is significant** (paired bootstrap): +0.066 / +0.044 at k = 3 / 8, and **+0.083 for small-object classes**. The gain holds within FAIR (0.215 → 0.310) and xView (0.127 → 0.167), so it is not an artifact of which source an image comes from.
- **For natural-image CLIP, crops do not help.** Local evidence requires an encoder that understands aerial imagery.
- **Crop count:** 13 crops perform as well as 29, and 4 are close, which keeps the selector cheap. GPU timing is pending.

### 7.5 M5 step 2 — do crops improve detection? (same 200 dev images, RTX 5060)
| RemoteCLIP selector, source-aware AP50 | 3 classes | 8 classes | 20 classes |
|---|---|---|---|
| Whole image (M4) | 0.064 | 0.058 | 0.040 |
| Whole image + 13 crops | 0.060 | 0.053 | 0.044 |
| Crops only | 0.067 | 0.061 | 0.040 |

**What this shows:**
- No difference is statistically significant (per-class sign tests, p ≥ 0.16).
- With crops, small-object **recall** rises (0.102 → 0.123–0.132 at 8 classes), but small-object **AP falls** (0.035 → 0.021): the detector now looks for small objects but boxes and scores them poorly.
- Crops cost 83 ms per image on the GPU, against ~8 ms for the whole-image selector, which removes the speed advantage.
- The crop selector was reproduced exactly on the GPU machine from a clean commit.
- **Conclusion:** improving the vocabulary cannot fix small-object detection on its own. That limit sits in the detector, at this input resolution. The whole-image RemoteCLIP selector stays the method's base.

### 7.6 M6/M7 step 1 — smarter vocabulary sizes (200 held-out dev images, no detector)
Each rule's single setting was tuned on the 530 *other* dev images to hit a target average size, then compared with fixed top-k at the same size.

- **Uncertainty retention** (a core of top-classes, plus classes whose crop evidence beats the whole-image score) helps at small budgets. At ~3 classes it keeps more of the present classes (0.217 → 0.266, significant), and small-object classes nearly triple (0.071 → 0.190). The gain fades at 8 classes and reverses at 20.
- **Adaptive budgets** (a calibrated threshold, a probability-mass rule) do **not** beat fixed top-k at the same average size.
- **Calibrated coverage roughly holds** (target 80% → 76% achieved), but needs ~27 classes per image, so the per-image relevance signal is still weak.
- Detection with these rules runs next.

### 7.7 M6/M7 step 2 — detection with retention and adaptive budgets (same 200 dev images, RTX 5060)
| Selector | Average classes | AP50 standard | AP50 source-aware | Robust vs full (both protocols)? |
|---|---|---|---|---|
| Full vocabulary | 80 | 0.015 | 0.028 | – |
| Fixed top-8 (M4) | 8 | 0.035 | 0.058 | source-aware only |
| **Calibrated threshold** | ~8 | **0.055** | **0.071** | **yes** |
| Calibrated threshold | ~20 | 0.047 | 0.069 | yes |
| Probability-mass rule | ~20 | 0.049 | 0.067 | yes |
| Uncertainty retention (crop disagreement) | ~8 | 0.033 | 0.046 | no |

**What this shows:**
- **Against a fixed top-k** at the same average size, adaptive budgets gain +0.02–0.03 AP50 at 8–20 classes. That passes 2 of our 3 checks (all 3 only for the probability-mass rule at ~20 classes under standard scoring), so it is **suggestive but not established** by our pre-declared criterion. The test split will settle it.
- The gain comes mainly from **large objects** (AP50 0.088 → 0.128). Confident images get fewer, cleaner prompts, and ambiguous ones get more.
- Image-level recall had predicted *no* gain, a second case where selection metrics mislead.
- **Uncertainty retention did not help detection**, repeating the M5 pattern for small objects.
- **Caveat:** the headline rule was the best of 9 conditions on these images. It is now fixed and will be confirmed once on the held-out test split.

### 7.8 M9 diagnostic — would more pixels fix small objects? (same 200 dev images)
We ran Grounding DINO on 4 overlapping tiles plus the full image, giving small objects about 1.6× more pixels.

| Vocabulary | AP50 standard, normal → tiled | Small-object AP50 | Time per image |
|---|---|---|---|
| All 80 classes | 0.015 → 0.013 | 0.012 → 0.009 | 0.42 → 2.1 s |
| Oracle | 0.140 → 0.114 | 0.100 → 0.093 | 0.34 → 1.7 s |
| Adaptive threshold (ours) | 0.055 → 0.039 | 0.036 → 0.020 | 0.34 → 1.7 s |

**What this shows:**
- Tiling slightly raises small-object recall but **not** small-object accuracy.
- It cuts large objects at tile borders and is 5× slower.
- **The small-object weakness is a recognition (domain) problem of the detector, not a resolution problem.** It needs an aerial-adapted detector, which is outside this study, so it is reported as a limitation.
- Vocabulary selection still helps on top of tiling (+0.025 AP50).

Full tables: [`generated/E0_tables.md`](generated/E0_tables.md), [`generated/e0_headroom_tables.md`](generated/e0_headroom_tables.md) (robustness), [`generated/M4_selection_tables.md`](generated/M4_selection_tables.md), [`generated/m4_selectors_det_tables.md`](generated/m4_selectors_det_tables.md), [`generated/M5_selection_tables.md`](generated/M5_selection_tables.md), [`generated/m5_global_local_det_tables.md`](generated/m5_global_local_det_tables.md), [`generated/M6_retention_tables.md`](generated/M6_retention_tables.md), [`generated/m6_retention_det_tables.md`](generated/m6_retention_det_tables.md) (+ `_vs_k3/_vs_k8/_vs_k20`).

---

## 8. Comparison plan (experiment matrix)
| ID | Method | Deployable | Status |
|---|---|---|---|
| E1 | Full vocabulary | ✓ | done (E0) |
| E2 | Fixed reduced vocabulary (frequency prior) | ✓ | done: worse than full |
| E3 | Random vocabulary (3 seeds) | ✓ (control) | done: worse than full |
| E4 | Global CLIP / RemoteCLIP top-k | ✓ | done: **robustly better than full at k = 8–20** (source-aware) |
| E5 | Caption-based ("VocAda-style", small captioner) | ✓ | planned |
| E6 | **Global + local (multi-scale crops) selection** | ✓ | done: better selection (small objects), **no detection gain**, 83 ms selector cost (vs ~8 ms global) |
| E7 | **+ uncertainty-aware retention** (global–local disagreement; calibrated on held-out dev images) | ✓ | done: selection gain at ~3 classes, **no detection gain** |
| E8 | **+ adaptive vocabulary budget** (threshold, probability mass) | ✓ | done: **best method so far**: AP50 ×3.8 (standard) / ×2.5 (source-aware) vs full, robust; +0.02–0.03 vs fixed top-k |
| — | Oracle diagnostics (GT classes, source class list) | ✗ | done (E0); used as upper bounds |

Each method is compared at **matched average vocabulary sizes**, with accuracy–latency curves, recall broken down by object size and class rarity, both evaluation protocols, and paired bootstrap confidence intervals.

## 9. Next steps
1. **Remaining baselines:** a caption-based (VocAda-style) selector and a simplified TagCLIP-style selector.
   - *Decided at M9:* whether a detector-side small-object test (tiled / higher-resolution inference, applied identically to every method) is needed. Until then the scope stays on vocabulary selection.
2. **E5:** the caption-based (VocAda-style) baseline, with a small captioner.
3. **M10–M11:** failure analysis (tiny objects, dense scenes, similar classes, rare classes) and per-source generalization. One-time final evaluation on the held-out test split.
4. **M12–M14:** latency profiling, reproducibility package, paper draft.

## 10. Risks and honest assessment
- **Detector ceiling:** GDINO-T reaches only 0.14 AP50 even with a perfect vocabulary, so absolute numbers will be low. Higher input resolution or tiling may be needed, applied identically to all methods.
- **Effect size:** the robust gains so far are modest in absolute terms (+0.02–0.03 AP50) because the detector itself is weak on aerial imagery. Under standard COCO scoring they are not yet robust.
- **Statistics:** percentile bootstrap intervals are biased for class-averaged AP, so every claim is now checked three ways. Detection experiments so far use 200 dev images; final claims will use the held-out test split once, at the end.
- **Publication realism:** with a frozen GDINO-T and one benchmark, a main-track CVPR/ICCV/ECCV paper is unlikely unless effects are large and consistent. A strong workshop paper (e.g. CVPR EarthVision) or IGARSS / IEEE GRSL / JSTARS is realistic. No acceptance is assumed.

## References (primary sources)
1. S. Liu et al., *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection*, ECCV 2024 (arXiv 2303.05499).
2. M. Liu, T. L. Hayes, M. Mancini, E. Ricci, R. Volpi, G. Csurka, *Test-Time Vocabulary Adaptation for Language-Driven Object Detection*, ICIP 2025 (arXiv 2506.00333).
3. C. Tsourveloudis, *Do Open-Vocabulary Detectors Transfer to Aerial Imagery? A Comparative Evaluation*, arXiv 2601.22164, 2026.
4. J. Pan et al., *Locate Anything on Earth: Advancing Open-Vocabulary Object Detection for Remote Sensing Community*, AAAI 2025 (arXiv 2408.09110).
5. Y. Li et al., *Toward Open Vocabulary Aerial Object Detection with CLIP-Activated Student-Teacher Learning*, ECCV 2024 (arXiv 2311.11646).
6. Z. Huang et al., *OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images*, ICCV 2025 (arXiv 2503.06146).
7. X. Zhao et al., *Scene-adaptive and Region-aware Multi-modal Prompt for Open Vocabulary Object Detection*, CVPR 2024.
8. Y. Lin et al., *TagCLIP: A Local-to-Global Framework to Enhance Open-Vocabulary Multi-Label Classification of CLIP Without Training*, AAAI 2024 (arXiv 2312.12828).
9. A. Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021.
10. F. Liu et al., *RemoteCLIP: A Vision Language Foundation Model for Remote Sensing*, IEEE TGRS 2024 (weights: HF `chendelong/RemoteCLIP`).
