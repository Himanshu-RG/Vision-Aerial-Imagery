# E5 — Caption-based (VocAda-style) and TagCLIP-style baselines (complete, 2026-10-31)

Both are **labelled baselines, not reproductions** (D38).

## Caption-based (VocAda-style)
1. **Caption:** Florence-2-base (`florence-community/Florence-2-base`, sha256-locked), task `<MORE_DETAILED_CAPTION>`, beams 3, ≤ 128 new tokens, no sampling. VocAda used LLaVA-Next-7B, which does not fit the 8 GB GPU.
2. **Phrases:** all 1–3-grams of consecutive non-stopword tokens. VocAda used spaCy noun phrases; we avoided a new dependency on both machines.
3. **Class score:** max over phrases of the OpenAI CLIP ViT-B/32 text–text cosine. VocAda used CLIP ViT-L/14 text similarity.
4. **Selection:** top-k, plus a threshold calibrated on the 530 calibration images to mean size 3 / 8 (D32).

All 730 dev images were captioned with 0 failures and 0 empty captions (mean 36.2 phrases per caption). CPU time was ~5.6 s per image. On the RTX 5060 (50 images, `outputs/selection_gpu/dev/captions_gpu_timing.json`) the median is **1,170 ms per image** (p90 1,322 ms; the first image, 6.6 s, is warm-up). 49/50 GPU captions are identical to the CPU ones. The captions are committed in `outputs/selection/dev/captions_florence2_base.json`.

## TagCLIP-style (crop-level analogue)
OpenAI CLIP, local-only, max over all 29 crops (M5 variant `a0_max_g234`). TagCLIP itself classifies patch tokens with attention refinement. This keeps only its local-to-global "max over local evidence" idea.

## Selection quality (200 held-out dev images)
Source: `reports/generated/E5_selection_tables.md`.

| k | RemoteCLIP (ours, base) | caption (VocAda-style) | TagCLIP-style |
|---|---|---|---|
| 3 | 0.217 / 0.295 | 0.181 / 0.115 | 0.174 / 0.223 |
| 8 | 0.388 / 0.508 | 0.355 / 0.212 | 0.338 / 0.375 |
| 20 | 0.662 / 0.697 | 0.533 / 0.411 | 0.630 / 0.631 |

Each cell is image vocabulary recall / macro (class-averaged) pair recall. Caption − RemoteCLIP image recall: −0.037 (k=3, n.s.), −0.033 (k=8, n.s.), **−0.129 (k=20, CI [−0.202, −0.053])**.

### Findings
- **The caption selector is comparable on image-level recall at small k but much worse on class-averaged recall**, less than half of RemoteCLIP's at k = 3/8. It repeatedly proposes the same generic classes, and a natural-image captioner rarely names aerial-specific categories (e.g. windmill and train-station images are described as "desert landscape" and "city"). Class-averaged recall predicted detection AP in M4–M6 (D27), so a weaker detection result is expected.
- **TagCLIP-style is below RemoteCLIP at every k.** Natural-image CLIP on crops is weak (cf. M5).
- **Cost:** captioning takes ~1.2 s/image on the RTX 5060 (~5.6 s on CPU). That is ~3.4× the detector forward pass itself and ~14× our crop-based selector (83 ms). VocAda reports 5–11 s/image on a V100 with a 7B captioner.

## Detection (RTX 5060, 200 dev images)
**Run.** `e5_caption_tagclip_det-69e357c3b1f4`: code `222905e` (clean), GPU idle at start (1,668 MB, 0%), 11 conditions, 0 failures. Pushed by Amit as `37637e7`. The shared conditions reproduce earlier runs exactly: full 0.0146 / 0.0282, oracle 0.1404, RemoteCLIP k8 0.0349 / 0.0582, headline 0.0550 / 0.0707.
Tables:
- `reports/generated/e5_caption_tagclip_det_tables.md` (vs full)
- `reports/generated/e5_caption_tagclip_det_tables_vs_rc8.md` (vs RemoteCLIP top-8)

| condition | size | AP50 std | AP50 src-aware | vs full (D26 robust?) | vs RemoteCLIP top-8 |
|---|---|---|---|---|---|
| full | 80 | 0.0146 | 0.0282 | ref | worse |
| RemoteCLIP top-8 | 8 | 0.0349 | 0.0582 | robust (src-aware) | ref |
| **headline (threshold m≈8)** | 7.8 | **0.0550** | **0.0707** | **robust (both)** | +0.020 / +0.013, not robust (sign test 33/11, p = 0.001 std) |
| caption top-3 | 3 | 0.0251 | 0.0324 | n.s. | lower (sign test 9/34, p = 0.0002) |
| caption top-8 | 8 | 0.0216 | 0.0335 | n.s. | **robustly worse (src-aware)** |
| caption threshold m≈3 | 2.9 | 0.0270 | 0.0389 | n.s. | lower (sign test 11/33, p = 0.001) |
| caption threshold m≈8 | 8.0 | 0.0186 | 0.0347 | n.s. | lower (sign test p = 0.03 src-aware) |
| TagCLIP-style top-3 | 3 | 0.0229 | 0.0460 | n.s. | lower (sign test p = 0.02–0.04) |
| TagCLIP-style top-8 | 8 | 0.0225 | 0.0525 | robust (src-aware only) | lower, n.s. |

### Findings
1. **Neither baseline beats the full vocabulary on AP50 by the D26 rule**, except TagCLIP-style top-8 under source-aware scoring. Every caption variant is below RemoteCLIP top-8 by the per-class sign test (more classes worse than better), and caption top-8 is robustly worse under source-aware scoring. As predicted from macro pair recall (D27).
2. **The headline method has the highest AP50 of every deployable selector tested** (E2–E8). Against the strongest baseline at the same size, RemoteCLIP top-8, it is +0.020 (std) / +0.013 (src-aware) with 33 classes better and 11 worse. That passes 1–2 of 3 checks: supportive, not robust.
3. **Mixed result: the caption selector wins on pooled P/R at fixed thresholds.**
   - At the pre-declared threshold 0.25, caption threshold m≈8 has 103 true positives against 37 for the headline method. F1@0.25 is 0.033 vs 0.013 (std) and 0.039 vs 0.014 (src-aware); best-F1 is 0.070 vs 0.032.
   - **Cause:** P/R/F1 are pooled over all boxes, so they are dominated by dense classes. Captions often mention cars, so the caption selector keeps FAIR1M's dense vehicle classes (Small Car AP50 0.126 vs 0.048, Van 0.096 vs 0.015). Class-averaged AP weights every class equally. There the headline method wins on rare and large structures that captions rarely name: stadium 1.00 vs 0.10, airplane 0.99 vs 0.56, windmill 0.22 vs 0, dam 0.20 vs 0, expressway toll station 0.30 vs 0.
   - Absolute numbers are tiny (103 of 4,788 GT boxes found at 0.25), so this does not change the AP-based conclusion. It is **reported, not hidden**: the methods' advantages are complementary, and the ranking depends on the metric. AP remains the pre-declared headline metric (M9).
   - A caption + RemoteCLIP union is an obvious idea suggested by this. It was **not** run: it would be a post-hoc method chosen after seeing dev results. If tried, it must be pre-declared and judged on the held-out test once.
4. **Cost:** caption selection adds ~1.2 s/image on the GPU, against 345 ms for detection. TagCLIP-style uses the same crops as M5 (83 ms).

**Conclusion for E5:** the two closest prior-work-style selectors do not beat our RemoteCLIP-based adaptive selector on the headline metric. The caption selector is 14× slower, and better only on pooled box-level F1, which dense vehicle classes drive.
