# M5 — Global–Local Category Relevance (steps 1–2 done, 2026-10-30)

## Objective
Test whether **local (crop-level) image evidence** recovers categories that a single **global** image embedding misses, especially small-object categories (**H3**). The global selector is the M4 selector.

## Method (fixed before results; tests in `tests/test_selectors.py`)
- **Views:** the whole image plus overlapping crops from a 2×2, 3×3 and 4×4 grid (4 + 9 + 16 = 29 crops).
  - Each crop is 25% larger than its grid cell, so it overlaps its neighbours.
  - Crop geometry depends only on the decoded image (rule D16 holds).
- **Per-view class probabilities:** softmax over the 80 classes of `logit_scale × cosine(view, class-text)`. The text uses the same three templates as M4 (D23), and the model's own logit scale (RemoteCLIP 99.9, CLIP 100.0). This puts global and local evidence on one scale. CLIP similarities are not calibrated probabilities.
- **Combination:** score(c) = α·p_global(c) + (1−α)·AGG_crops p_crop(c). Variants compared:
  - α ∈ {0, 0.25, 0.5, 0.75, 1};
  - AGG ∈ {max, mean};
  - crop grids ∈ {2×2, 2×2+3×3, 2×2+3×3+4×4}.

  That gives 25 variants per model. α = 1 is the M4 global selector; α = 0 is local-only.
- **Selection rule (D28, fixed before results):** per model, take the variant with the highest class-wise ranking mAP on dev. Ties go to fewer crops, then higher α, then max. This metric was chosen because it ordered the M4 detection results correctly (D27).
- **Code and outputs:**
  - Code: `src/aerial_ovd/vocabulary/selectors.py` (`crop_boxes`, `view_logits`, `combine_views`), `scripts/run_views.py`, `scripts/m5_global_local.py`, `scripts/compare_views.py`.
  - Tables: **`reports/generated/M5_selection_tables.md`**. Figure: `outputs/figures/m5_recall_dev.png`. Metrics: `outputs/metrics/m5_selection_dev/m5_metrics.json`.

## Step 1 — selection quality (730 dev images, CPU, detector-free)
Both models: 730/730 images processed, 0 failures, 29 crops per image. The CPU took ~0.9 s per image for all 30 views (not a latency result).

| model | variant chosen by D28 | crops used | ranking mAP global → chosen | vocab recall k=3 / 8 / 20 (global → chosen) | small-object pair recall k=8 |
|---|---|---|---|---|---|
| **RemoteCLIP** | `a0.25_mean_g23` (α = 0.25, mean, 2×2+3×3) | 13 | 0.245 → **0.269** | 0.221 → **0.287** / 0.410 → **0.455** / 0.668 → **0.689** | 0.222 → **0.305** |
| OpenAI CLIP | `a0.5_max_g23` (α = 0.5, max, 2×2+3×3) | 13 | 0.167 → 0.192 | 0.189 → 0.186 / 0.356 → 0.343 / 0.609 → 0.638 | 0.099 → 0.094 |

Paired image-level bootstrap (2,000 resamples, 95% CI), chosen minus global:
- **RemoteCLIP:**
  - vocabulary recall k=3 **+0.066 [+0.046, +0.085]**; k=8 **+0.044 [+0.024, +0.065]**; k=20 **+0.021 [+0.002, +0.042]**;
  - small-object pair recall k=8 **+0.083 [+0.060, +0.106]**.
- **CLIP:**
  - k=3 −0.003 (n.s.); k=8 −0.013 (n.s.); k=20 +0.028 [+0.005, +0.051];
  - small-object pair recall −0.005 (n.s.).

Small-object pair recall at k=8 by source (RemoteCLIP, global → chosen):

| source | global | chosen |
|---|---|---|
| FAIR | 0.215 | **0.310** |
| xView | 0.127 | **0.167** |
| DIOR | 0.917 | 0.833 |

DIOR has few small-object pairs.

### Findings
1. **H3 is supported at the selection level, for the remote-sensing encoder.**
   - Adding crop evidence to RemoteCLIP raises recall of small-object (image, class) pairs by +0.083 at k = 8 (about ×1.4), and by about ×2 at k = 3 (figure).
   - It also raises overall vocabulary recall at every budget tested.
   - The gain holds **within** FAIR and xView, so it is not just the size–source confound noted in M4.
2. **Local evidence only helps with an aerial-adapted encoder.** For natural-image CLIP, crops do not improve recall at small budgets or small-object recall. Plausibly CLIP cannot recognise aerial objects in low-context crops, while RemoteCLIP was trained on remote-sensing imagery. This was not anticipated, and it is useful evidence for the paper.
3. **Mean beats max for RemoteCLIP.** Its best variants all use mean aggregation with a low global weight (α = 0.25). Max aggregation lets one spurious crop dominate.
4. **More crops are not better.** 13 crops (2×2+3×3) match 29 crops (ranking mAP 0.269 vs 0.267), and even 4 crops (2×2) are close (0.267). That matters for cost.
5. **Ranking-mAP values differ from M4.** M4 ranked raw cosines; M5 ranks per-image softmax probabilities. Within an image the order is identical, so the global top-k sets are unchanged. Verified on all 730 dev images: identical top-k sets at k = 3, 8, 20 for both models. Across images the normalisation changes class-wise ranking, so the global ranking mAP here is 0.245 (RemoteCLIP) vs 0.179 in M4. Compare M5 variants with each other, not with M4 numbers.

### Caveats
- **Which statistical rule applies.** The step-1 CIs are percentile bootstraps of **means of per-image quantities** (vocabulary recall; small-pair recall as a ratio of image-level sums). Those are smooth statistics, for which the percentile interval is appropriate. The bias that led to D26 affects **class-averaged AP**, where classes drop out of resamples. D26 therefore governs the detection-level comparisons, including M5 step 2.
- **Selected on dev, evaluated on dev.** D28 picks among 25 variants using dev, so the chosen variant's dev numbers are mildly optimistic. The final, unbiased check is the one-time test evaluation. The top RemoteCLIP variants are within 0.005 ranking mAP of each other, so the choice is not fragile.
- **Provenance:** the views were computed with selector code that was committed only afterwards (meta `git_dirty_code` true; the code did not change between the run and the commit). Amit's GPU run on a clean commit, checked with `scripts/compare_views.py`, provides the independent reproduction.
- CPU timing is not a latency result. GPU timing of the 14 views the chosen variant uses is pending; the H6 budget is ≲ 85 ms/image.

## Step 2 — detection (RTX 5060)
**Run.**
- `m5_global_local_det-aab45f29ca9b`, code `0a6ca79` (clean), the same 200 dev images as E0/M4, 16 conditions, 0 failures.
- GPU idle at start (1,175 MB, 0% util). Pushed by Amit as `233f231`.
- Tables: **`reports/generated/m5_global_local_det_tables.md`** (vs `full`).
- `full`, `oracle_gt` and the global selectors reproduce E0/M4 exactly (e.g. RemoteCLIP k8, source-aware AP50 0.0582).

**Independent reproduction of the crop selector.** On a clean commit, the GPU views (grids 2×2+3×3) reproduce the committed `*_gl` selectors: identical top-k sets at k = 3/8/20 for both models; max |score diff| 3.4×10⁻⁶ (RemoteCLIP) and 1.2×10⁻⁵ (CLIP). This resolves the step-1 provenance caveat.

**Global+local vs global at the same k.**
- The raw predictions live on the RTX 5060, so a re-referenced bootstrap was not run.
- The comparison uses point differences plus the per-class sign test (D26 check iii), computed from `metrics.json` by `scripts/compare_conditions.py`.
- Full table, including per-source AP50 (FAIR/xView/DIOR/DOTA): **`reports/generated/m5_gl_vs_global.md`**.
- Every paired difference below is **not significant** (all sign-test p ≥ 0.16).

| RemoteCLIP, source-aware AP50@900 | k=3 | k=8 | k=20 |
|---|---|---|---|
| global (M4) | 0.0643 | 0.0582 | 0.0404 |
| global + local (chosen, D28) | 0.0602 | 0.0533 | 0.0444 |
| local only | 0.0666 | 0.0607 | 0.0395 |
| classes better / worse, GL vs global | 14 / 20 | 17 / 26 | 29 / 21 |

| RemoteCLIP, small objects (standard) | k=3 | k=8 | k=20 |
|---|---|---|---|
| R50 small, global → GL → local | 0.071 → 0.093 → 0.053 | 0.102 → 0.123 → 0.132 | 0.158 → 0.180 → 0.197 |
| AP50 small, global → GL | 0.021 → 0.008 | 0.035 → 0.021 | 0.022 → 0.021 |

For CLIP, global+local vs global gives k=8: 0.0461 → 0.0528 and k=20: 0.0492 → 0.0418 (both n.s.).

### Findings
1. **H3 is NOT supported at the detection level.** Global+local selection raises *selection* recall of small-object classes (step 1), and the detector does find more small objects. Small-object R50 rises at k = 8/20 (e.g. 0.102 → 0.123/0.132 at k = 8). But **small-object AP50 does not rise; it falls** (0.035 → 0.021 at k = 8), and overall AP50 is unchanged (no significant difference at any k).
2. **The bottleneck moves from vocabulary to the detector.** Once the small-object class names reach Grounding DINO, it detects more of those objects but localises and scores them poorly. The newly admitted classes also add false positives. This matches E0: even with the oracle vocabulary, small-object AP50 is only 0.100 and 18/75 classes stay near 0. Better selection alone cannot fix small-object *detection* with GDINO-T at 800/1333 px.
3. **Cost cancels the latency gain.** The crop selector needs **83 ms/image** on the RTX 5060 (RemoteCLIP, 14 views; CLIP 78 ms), vs ~8 ms for the global selector. End-to-end at k = 8 that is ≈ 360 + 83 = 443 ms vs 453 ms for `full`, so essentially no speed-up (the global selector gives ≈ 368 ms).
4. **Implication for the method.** The cheaper **global RemoteCLIP selector remains the best deployable selector** for detection. Local evidence is still valuable as a *diagnostic* (it shows which classes were missed), and possibly as an input signal for uncertainty-aware retention (M6), but not as a direct replacement.

Per source (source-aware AP50, RemoteCLIP k=8, global → GL): FAIR 0.011 → 0.014, xView 0.008 → 0.004, DIOR 0.152 → 0.141, DOTA 0.292 → 0.265. There is no consistent gain even in the small-object-heavy sources.

**GPU-state caveat:** `run_config.json` records 1,175 MB in use and 0% utilisation at the start. Amit's commit notes that the display runs on the RTX 5060 (~1.4–1.9 GB in use), so latencies include that background load. H6 timings stay provisional until re-measured on an idle GPU (M12).

### Recommended follow-ups
- **M6/M7:** build on the global RemoteCLIP selector. Use global–local *disagreement* as a candidate uncertainty signal, which the master prompt lists explicitly.
- **Detector-side small-object test (later, M10/M12):** tiled / higher-resolution inference, applied identically to every condition. This would test whether small-object classes, once selected, become detectable.

## Exit criteria (M5)
| Criterion | Status |
|---|---|
| Working global–local selector | ✅ |
| Ablation: global vs local vs global+local (selection level) | ✅ |
| Measurable benefit of local evidence | ✅ selection level (RemoteCLIP); ❌ detection level (no AP gain; recall up, precision down) |
| Selector overhead (crop count, timing) | ✅ 83 ms/image (RemoteCLIP, 14 views, RTX 5060) vs ~8 ms global |
| TagCLIP-style baseline | ⏳ not implemented yet (a simplified, clearly labelled version is planned) |
