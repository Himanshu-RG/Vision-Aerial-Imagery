# Adaptive Vocabulary Selection for Open-Vocabulary Object Detection in Aerial Imagery

*Draft v0.1 (2026-11-02), M14. Target: a remote-sensing workshop (e.g. CVPR EarthVision) or IGARSS / IEEE GRSL. Every number below comes from committed metrics; see `REPRODUCE.md`. Items marked **[TODO]** must be resolved before submission.*

**Authors:** Amit Adhikari, Himanshu Ray, Saurav Kumar Gupta, Avishek Chowdhury (Team Transmontane). **[TODO: affiliation, supervisor]**

---

## Abstract
Open-vocabulary detectors such as Grounding DINO detect any class named in a text prompt. On aerial imagery, however, prompting with a long list of fine-grained, overlapping class names makes accuracy collapse. We ask whether a **training-free, per-image vocabulary selector** can recover part of this loss for a **frozen** detector.

Our selector scores the 80 candidate classes with the remote-sensing CLIP model RemoteCLIP. It keeps every class whose probability exceeds a threshold, calibrated once on held-out images, so the vocabulary size adapts to each image (about 8 classes on average). It sees only pixels and costs about 8 ms per image.

**Evaluation.** We evaluate on LAE-80C with a source-aware protocol that accounts for its five disjoint label spaces. A difference counts as robust only if three statistical checks agree, and all method choices were fixed before a single evaluation on 2,862 held-out test images.

**Results.**
- The selector raises AP50 of Grounding DINO-T from 0.0053 to 0.0280 (standard COCO) and from 0.0157 to 0.0418 (source-aware), robust under both protocols.
- It also beats a fixed-size top-8 selector with the same encoder.
- The gain is confined to imagery and label spaces resembling the encoder's training data. It is ×3.5 on DIOR/DOTA, transfers untuned to NWPU VHR-10 (0.088 → 0.129), and is absent on FAIR1M and xView.
- Local (crop) evidence, uncertainty-aware retention, caption-based selection and synonym expansion do not improve detection.
- Failure analysis shows the remaining limits lie in cross-dataset near-synonym classes and in the detector itself.

## 1. Introduction
**The problem.**
- Open-vocabulary (OV) detectors ground a free-form text prompt in the image. With the prompt fixed, the user decides which classes to look for.
- In aerial imagery the natural choice, listing every class of interest, works poorly. Benchmarks such as LAE-80C have 80 classes, many of them near-duplicates (`vehicle`, `small car`, `passenger car`, `van`).
- A recent zero-shot study on LAE-80C [3] found that prompting with only the ground-truth classes of each image raises F1 by 15×. It used the ground truth, however, so it is an upper bound rather than a method.

**Test-time vocabulary selection.**
- Choosing the prompt per image from the image itself has been studied for natural images. VocAda [2] uses a 7B captioner and other detectors, at 5–11 s per image.
- To our knowledge, no prior work studies deployable per-image selection for a frozen OV detector on aerial imagery. Our citation sweep (2026-10) found none. **[TODO: re-check before submission]**

We study this question under a strict protocol and report what works and what does not. **Contributions:**
1. **A deployable, training-free adaptive selector.** RemoteCLIP probability with a calibrated per-image threshold. Pixels only, about 8 ms per image, no detector change.
2. **An evaluation that accounts for merged-benchmark label spaces.** We show LAE-80C's five sources annotate disjoint class sets, and we score source-aware alongside standard COCO. All claims require agreement of percentile and basic bootstrap intervals and a per-class sign test.
3. **A pre-registered held-out test.** The selector improves AP50 ×5.3 (standard) and ×2.7 (source-aware). It also beats fixed-size selection with the same encoder.
4. **A map of where it works and why it stops.**
   - It works on imagery like the encoder's training data (DIOR/DOTA, and NWPU VHR-10 untuned), but not on fine-grained, unfamiliar label spaces.
   - Four plausible extensions gave negative results: crops, retention, captions, synonyms.
   - The failure analysis attributes the remaining gap to cross-source near-synonyms and to detector limits on small objects.

## 2. Related work
**Open-vocabulary detection.**
- Grounding DINO [1] fuses image and text features and scores boxes against prompt tokens. We use its Swin-T variant, frozen.
- Aerial OV detectors (LAE-DINO [4], CastDet [5], OpenRSD [6]) train on remote-sensing data. LAE-DINO builds its vocabulary dynamically during *training*.
- Our approach is complementary: it changes only the prompt at inference.

**Prompt / vocabulary adaptation.**
- Scene-adaptive prompts [7] learn prompt modifications.
- VocAda [2] selects classes per image via a captioner and CLIP or an LLM, on COCO/Objects365 with Detic-style detectors.
- TagCLIP [8] tags images by aggregating local CLIP evidence.
- We evaluate VocAda-style (caption) and TagCLIP-style (crop) selectors as baselines. They are labelled "-style": simplified to fit an 8 GB GPU, not reproductions.

**Zero-shot aerial evaluation.** Tsourveloudis [3] evaluates five OV detectors on LAE-80C. It reports the oracle-vocabulary gain and calls for "intelligent vocabulary filtering". We use the same benchmark with COCO AP, a deployable selector and a held-out test.

## 3. Method
**Setup.** Given an image *I* and a master vocabulary *V* of |V| = 80 class names, a selector returns a subset *S(I) ⊆ V*. The frozen detector is then prompted with `"s1 . s2 . … ."` in the official caption format. Class scores are the mean token sigmoid over each name, and the top 900 (box, class) pairs are kept.

**Relevance scores.**
- Each class *c* gets a text embedding *t_c*: the normalised mean of three fixed templates ("a satellite image of a {}.", "an aerial photo of a {}.", "a photo of a {}."), fixed a priori.
- The image is encoded by RemoteCLIP ViT-B/32 [10] at 224 px, giving *e_I*.
- The class probability is p(c | I) = softmax_c(λ · ⟨e_I, t_c⟩), with the model's own logit scale λ = 99.85.

**Adaptive selection.**
- *S(I) = {c : p(c | I) ≥ t}*. At least 1 and at most 40 classes are kept.
- One parameter, the threshold *t*, is set by bisection on 530 calibration images so that the **mean** vocabulary size is 8 (*t* = 0.0237). It is never tuned to a detection score.
- Confident images receive few classes, ambiguous ones more (1–17 on dev).
- **Input rule:** the selector receives only pixels, never file names, image sizes or source tags. Image size alone identifies LAE-80C's source datasets. A unit test enforces this.

**Variants tested and not adopted** (all pre-declared, dev only):
- (i) global + local evidence from 13 overlapping crops;
- (ii) uncertainty-aware retention of classes where crop and global evidence disagree;
- (iii) a probability-mass (top-p) budget;
- (iv) synonym expansion by class-name text similarity;
- (v) VocAda-style caption selection (Florence-2-base) and a TagCLIP-style crop maximum.

## 4. Experimental protocol
**Data.** LAE-80C [4] has 3,592 images, 86,539 boxes after logged cleaning, and 80 classes from FAIR1M, xView, DIOR, DOTA and a power-plant set. We split it once into dev (730) and test (2,862) by iterative multi-label stratification (seed 2026) and committed the ID lists before any detection.
- Dev is used for all development: a 530 / 200 split for calibration vs detection.
- Test was used **once**, with every condition, parameter and analysis committed beforehand.
- NWPU VHR-10 (614 images, 9 classes mapped to LAE-80C names; ship dropped) is an external set. Every method is applied frozen.

**Disjoint label spaces.**
- Each LAE-80C source annotates only its own classes. A car is "Small Car" in FAIR1M, "Passenger Car" in xView and "vehicle" in DIOR.
- Standard COCO evaluation therefore counts a correct "vehicle" box on a FAIR1M image as a false positive.
- We report **source-aware** AP, which scores only classes in the image's source label space, alongside standard AP. Source identity is used only by the evaluator.

**Metrics and statistics.**
- The headline metric is AP50 at 900 detections per image (images hold up to 668 objects). We also report P/R/F1 at a score threshold fixed on dev.
- Differences use a paired image-level bootstrap (1,000 resamples).
- The percentile interval is biased for class-averaged AP, so a difference is called **robust** only if the percentile CI, the basic CI and a per-class sign test (p < 0.05, same direction) all agree.

**Implementation.**
- HF `transformers` Grounding DINO-T (172 M parameters), fp32, 800 px short side.
- `open_clip` RemoteCLIP / OpenAI CLIP ViT-B/32. All checkpoints are pinned by commit and sha256.
- Detection on an RTX 5060 (8 GB); selection scores computed on CPU and committed, so both machines use identical vocabularies.

## 5. Results
### 5.1 Headroom (dev, 200 images)
| Vocabulary | Classes | AP50 std | AP50 src-aware |
|---|---|---|---|
| all 80 classes | 80 | 0.015 | 0.028 |
| oracle: the image's source label space | ~21 | 0.031 | 0.031 |
| **oracle: classes present** | ~2 | **0.140** | **0.140** |
| random, size-matched | ~2 | 0.007 | 0.008 |

- The per-image oracle gains +0.112 source-aware (robust). About 89% of the gap survives source-aware scoring, so it reflects per-image relevance rather than an evaluation artifact.
- Giving the detector the right *dataset's* class list barely helps, and a random list of the right size does not help at all. The selector must find the classes actually present.

### 5.2 Development results (dev, 200 images)
| Selector | Avg. classes | AP50 std | AP50 src-aware |
|---|---|---|---|
| all 80 classes | 80 | 0.015 | 0.028 |
| frequency prior (in-sample) | 3–40 | – | 0.004–0.017 |
| VocAda-style caption top-8 | 8 | 0.022 | 0.034 |
| TagCLIP-style top-8 | 8 | 0.023 | 0.053 |
| OpenAI CLIP top-20 | 20 | – | 0.049 |
| RemoteCLIP top-8 | 8 | 0.035 | 0.058 |
| RemoteCLIP global + 13 crops, top-8 | 8 | – | 0.053 |
| uncertainty retention, ~8 | 8 | 0.033 | 0.046 |
| synonym-aware (E9), ~8 | 8 | 0.052 | 0.071 |
| **adaptive threshold, ~8 (ours)** | 7.9 | **0.055** | **0.071** |

**Negative results:**
- **Crops.** Local evidence improved *selection* of small-object classes (0.22 → 0.31 recall at k = 8), but not detection: small-object recall rose while small-object AP fell. Crops also cost 83 ms vs 8 ms per image.
- **Retention.** Uncertainty-aware retention repeated this pattern.
- **Captions.** The caption selector costs ~1.2 s per image and loses on class-averaged AP. It wins only on pooled F1, through dense vehicle classes.
- **Synonym expansion.** Its replacement rule was declared in advance, and it failed that rule.

Ours was chosen as the headline among 9 rule × size conditions on these images and frozen before the test.

### 5.3 Held-out test (2,862 images, one run)
| Selector | Avg. classes | AP50 std | AP50 src-aware | Ours vs this (3 checks) |
|---|---|---|---|---|
| all 80 classes | 80 | 0.0053 | 0.0157 | robust, both |
| random (oracle size) | 2.3 | 0.0023 | 0.0038 | robust, both |
| frequency prior top-8 | 8 | 0.0028 | 0.0056 | robust, both |
| OpenAI CLIP top-20 | 20 | 0.0081 | 0.0238 | robust (std) |
| RemoteCLIP top-8 | 8 | 0.0170 | 0.0347 | **robust, both** |
| adaptive threshold, ~20 | 19.8 | 0.0209 | 0.0378 | n.s. |
| adaptive top-p, ~20 | 19.8 | 0.0276 | 0.0406 | n.s. (tie) |
| **ours: adaptive threshold, ~8** | 8.0 | **0.0280** | **0.0418** | – |
| *oracle* | 2.3 | 0.1072 | 0.1072 | – |

**Primary claim (pre-declared).** Ours − all classes:
- **standard:** +0.0228, basic CI [+0.0143, +0.0263], 56 / 18 classes better / worse, p = 1.1e-5;
- **source-aware:** +0.0261, basic CI [+0.0163, +0.0300], 50 / 24, p = 0.0034.

Both are robust. This recovers 22% / 29% of the oracle headroom, against 38% on dev, the expected shrinkage after selection among dev conditions.

**Adaptive vs fixed size.** Against RemoteCLIP top-8, which uses the same encoder and the same mean size, ours is robustly better under both protocols: +0.011 standard (50/20 classes) and +0.007 source-aware (44/26). On dev this comparison passed only 2 of 3 checks.

**F1.**
- At the thresholds fixed on dev, F1 improves from 0.014 to 0.036 (standard) and from 0.039 to 0.056 (source-aware).
- A frequency prior reaches a higher pooled F1 (0.084 / 0.132) despite the lowest AP. Its fixed list consists mostly of FAIR1M vehicle classes plus xView "Building", the densest classes in the test set.
- We therefore keep the class-averaged AP as the headline, as pre-declared.

### 5.4 Where it works
| AP50 src-aware (test) | DIOR | DOTA | FAIR1M | xView | DIOR+DOTA | FAIR1M+xView+PP |
|---|---|---|---|---|---|---|
| all 80 classes | 0.039 | 0.022 | 0.010 | 0.006 | 0.035 | 0.008 |
| ours | 0.126 | 0.100 | 0.010 | 0.008 | **0.120** | **0.009** |
| oracle | 0.175 | 0.128 | 0.120 | 0.072 | 0.163 | 0.083 |

- RemoteCLIP's training data includes DIOR and DOTA, but not FAIR1M or xView [10]. The gain is entirely on the former, as already seen on dev and pre-declared for test, although the oracle shows ×11 headroom on the latter.
- Applied frozen to NWPU VHR-10, ours improves AP50 from 0.088 to 0.129 (+0.041, robust; oracle 0.208). NWPU was built by DIOR's authors with DIOR's class names, so this shows transfer to new images of a familiar kind. It is not a generic unseen-domain result.
- RemoteCLIP top-8 on NWPU gives no significant gain (0.093), which again points to the adaptive size.

### 5.5 Why it stops (failure analysis, dev)
- **Lost classes.** Ours keeps the right class for only 26% of ground-truth *objects*. Dense FAIR1M/xView scenes hold 85% of the objects.
- **Near-synonyms.** 59% of selected classes come from another source's label set (e.g. xView "Ferry" on a FAIR1M "Passenger Ship"). A diagnostic restricted to the right source raises object coverage 0.25 → 0.68. A deployable synonym-aware variant did not help.
- **Kept classes.** Where the right class is kept, the detector finds 48% of objects, against 58% with the oracle.
- **False positives.** Confident false positives drop by 76%: with all 80 classes, 77% of them are classes never annotated in that dataset.
- **Scene density.** Selection helps sparse scenes (1–5 objects: 0.025 → 0.109) but hurts scenes with more than 20 objects.
- **Small objects.** Tiled inference (2×2 + full image) did not raise small-object AP and cost 5× latency. Small objects are a detector limitation at this scale.

### 5.6 Cost
- The selector adds about 8 ms per image on the GPU. Shorter prompts reduce detector time.
- On test, the median detector time is 442 ms (80 classes) vs 347 ms (ours), about 20% faster end to end. That measurement was taken with ~2 GB of other GPU load.
- **[TODO: replace with the idle-GPU M12 numbers, `reports/generated/M12_latency.md`]**

## 6. Limitations
- **Low absolute accuracy.** The frozen GDINO-T reaches only 0.107 AP50 on test even with a perfect vocabulary, and 18 classes stay near zero on dev.
- **Scope of the gain.** It is limited to imagery and label spaces like the encoder's training data. NWPU–DIOR image overlap was not checked.
- **One detector, one benchmark family.** Results with larger detectors (GDINO-B, OWLv2) are untested.
- **Disclosed design choices.** The rule family and the 8-class target were designed by us on dev and frozen before test. The synonym experiment was motivated by the dev failure analysis.

## 7. Conclusion
- A per-image vocabulary chosen from pixels by an aerial CLIP model, with a calibrated adaptive size, robustly improves a frozen open-vocabulary detector on held-out aerial images, at negligible cost.
- The benefit is real but bounded. It holds where the selector's encoder knows the imagery and label space, and is capped by near-synonym classes across merged datasets and by the detector's weakness on small objects.
- Closing these gaps likely needs taxonomy-aware vocabularies and aerial-adapted detectors, rather than better selection alone.

## References
1. S. Liu et al., Grounding DINO, ECCV 2024 (arXiv 2303.05499).
2. M. Liu et al., Test-Time Vocabulary Adaptation for Language-Driven Object Detection (VocAda), ICIP 2025 (arXiv 2506.00333).
3. C. Tsourveloudis, Do Open-Vocabulary Detectors Transfer to Aerial Imagery?, arXiv 2601.22164, 2026.
4. J. Pan et al., Locate Anything on Earth (LAE-DINO, LAE-1M), AAAI 2025 (arXiv 2408.09110).
5. Y. Li et al., CastDet, ECCV 2024 (arXiv 2311.11646).
6. Z. Huang et al., OpenRSD, ICCV 2025 (arXiv 2503.06146).
7. X. Zhao et al., Scene-adaptive and Region-aware Multi-modal Prompt for OVD, CVPR 2024.
8. Y. Lin et al., TagCLIP, AAAI 2024 (arXiv 2312.12828).
9. A. Radford et al., CLIP, ICML 2021.
10. F. Liu et al., RemoteCLIP, IEEE TGRS 2024 (arXiv 2306.11029).
11. G. Cheng et al., NWPU VHR-10, ISPRS J. 2014. **[TODO: verify citation details]**

---
### Pre-submission checklist **[TODO]**
- [ ] M12 idle-GPU latency numbers into §5.6.
- [ ] Re-run the novelty sweep (VocAda and 2601.22164 citations since 2026-10-20).
- [ ] Figure 1: pipeline. Figure 2: `reports/figures/m10/selection_gain.jpg`. Figure 3: per-source bar chart (from `outputs/metrics/m11_per_source/final_test-75ab843910a3.json`).
- [ ] Venue template (LaTeX), page limit, anonymisation.
- [ ] Supervisor review.
