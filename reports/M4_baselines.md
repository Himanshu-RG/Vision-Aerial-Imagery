# M4 — Master Vocabulary & Simple Baselines (steps 1–2 done, 2026-10-25; baseline E deferred)

## Objective
Find out whether simple, deployable vocabulary selectors pick the classes that are present (step 1, detector-free), and whether that improves detection over the full 80-class vocabulary (step 2, GPU; tests **H2**).

## Baselines implemented
| ID | Selector | What it uses | Notes |
|---|---|---|---|
| A | `full` | nothing (all 80 classes) | reference |
| B | `frequency` | a fixed class prior: number of **dev** images per class | image-independent "fixed reduced vocabulary". Fit on dev, so **dev results are in-sample (optimistic)** |
| C | `random_s{0,1,2}` | seeded random scores (seed + image-content hash) | 3 seeds |
| D | `clip_openai_b32` | OpenAI CLIP ViT-B/32 (open_clip, `timm/vit_base_patch32_clip_224.openai` @ `a6f597a30f`) | global image–text cosine similarity |
| D′ | `remoteclip_b32` | RemoteCLIP ViT-B/32 (`chendelong/RemoteCLIP` @ `bf1d8a3ccf`) | remote-sensing CLIP. Its training captions derive from DOTA/DIOR **train** annotations (D7); LAE-80C uses val images |
| E | VocAda-style captioner | — | **not started** (needs a ~0.5–1 GB captioner; deferred, D8) |

- **Selection rule:** top-k by score, ties broken by lower class id (stable).
- **Pixels only (D16):** enforced by a test that checks the selector API takes only the image.
- **CLIP text prompts:** the mean of three fixed templates, chosen *a priori* and not tuned: "a satellite image of a {}." / "an aerial photo of a {}." / "a photo of a {}."
- **Text embeddings** depend only on the vocabulary, so they are cached once per run. Image embeddings are counted in selector time.
- **Verification:** both CLIP checkpoints are sha256-locked, and both were confirmed to load trained (distinct) weights. Sanity check: DIOR "train station" and "windmill" images rank correctly.

## Step 1 — selection quality on the full dev split (730 images, detector-free)
- Commands:
  - `python scripts/run_selection.py --split dev` (CPU, dev laptop; 0 failures for every selector);
  - `python scripts/evaluate_selection.py --split dev`.
- Outputs:
  - `outputs/selection/dev/*.npz` (scores, committed);
  - `outputs/metrics/selection_dev/selection_metrics.json`;
  - **`reports/generated/M4_selection_tables.md`**;
  - `outputs/figures/m4_vocab_recall_dev.png`.

Vocabulary recall (mean over images of |selected ∩ GT| / |GT|), from the generated tables:

| selector | k=1 | k=3 | k=5 | k=8 | k=12 | k=20 | k=40 | ranking mAP |
|---|---|---|---|---|---|---|---|---|
| frequency (in-sample) | 0.112 | 0.288 | 0.462 | 0.610 | 0.691 | 0.770 | 0.907 | 0.049 |
| CLIP ViT-B/32 | 0.101 | 0.189 | 0.268 | 0.356 | 0.446 | 0.609 | 0.864 | 0.144 |
| RemoteCLIP ViT-B/32 | 0.130 | 0.221 | 0.302 | 0.410 | 0.520 | 0.668 | 0.882 | 0.179 |
| random (3 seeds, mean) | 0.014 | 0.045 | 0.071 | 0.113 | 0.161 | 0.264 | 0.513 | – |

### Findings
1. **Global CLIP selection is weak at small budgets.** At k = 3, RemoteCLIP keeps 22% of the present classes (CLIP 19%); E0's oracle keeps 100% with ~2 classes. Even at k = 12, only about half are kept. Given E0's finding that missing classes are costly, small-k global selection is unlikely to beat the full vocabulary. Step 2 tests this directly.
2. **RemoteCLIP > CLIP at every budget below 80**, by +0.02 to +0.07 (largest at k = 12: 0.520 vs 0.446). Paired bootstrap was run for k = 1, 2, 3, 5, 8, and all those CIs exclude 0. Both reach 1.0 at k = 80 by construction.
3. **The image-independent frequency prior beats both CLIPs for k ≥ 2.** This is partly in-sample and reflects dataset bias: FAIR is 60% of dev, and its common vehicle classes dominate. Yet CLIP ranks *images* per class far better (ranking mAP 0.14–0.18 vs 0.05). So CLIP carries image-specific signal but mis-ranks classes *within* an image. A plausible reason, seen in spot checks: near-synonyms across sources such as `vehicle` (DIOR) vs `small car` (FAIR) vs `passenger car` (xView).
4. **Small objects are lost by global evidence.** At k = 5, the share of (image, class) pairs whose objects are mostly small (< 32² px) that get selected is **0.043 for CLIP and 0.155 for RemoteCLIP**, vs 0.435 / 0.354 for the other pairs.
   - This is direct, pre-registered-style evidence for the M5 motivation: global image embeddings under-represent small objects, so local (crop) evidence is needed.
   - Caveat: small-dominated pairs are mostly FAIR vehicles, so object size and source are confounded. M5 should break the analysis down by source.
5. **Cost:** ~50–70 ms per image for either CLIP on the laptop CPU (49–52 ms in one run, 69–71 ms in the next; CPU timing is noisy and not a latency result). GPU timing for H6 is pending on the RTX 5060. E0 implies a budget of ≲ 85 ms per image for a net latency win.

## Step 2 — detection with selected vocabularies (RTX 5060)
**Run.**
- `m4_selectors_det-16d38bd056f8` (config `configs/experiments/m4_selectors_detection.yaml`), on the RTX 5060 (`LAPTOP-5CPRTI27`).
- Code `913a99e`, clean tree. Pushed by Amit as `0111bee`.
- The **same 200 dev images as E0** (verified identical ID list). 17 conditions, **0 failures**.
- Selector inputs are the committed npz files (code `ee7afa1`, clean).
- Tables: **`reports/generated/m4_selectors_det_tables.md`** (generated by `scripts/report_run.py`).

**Reproducibility checks:**
- `full` and `oracle_gt` reproduce E0 exactly (AP50@900 0.0146 / 0.0282 / 0.1404).
- GPU-computed selector scores match the committed CPU scores (max |diff| 1.6×10⁻⁶; identical top-k sets at k = 3/8/20).

**Statistics.** The pre-registered percentile bootstrap CI turned out to be biased for class-averaged AP: the bias flag was raised for most comparisons, and the point estimates sit near the lower CI bound. We therefore report three checks and call a difference **robust** only if all three agree. **This robustness rule (D26) was adopted after the step-2 results were seen.** It is stricter than the pre-registered rule (percentile CI excludes 0), so H2 holds under either. The paper must state this.
- (i) the percentile CI excludes 0;
- (ii) the "basic" bootstrap CI excludes 0;
- (iii) a per-class paired sign test agrees in direction with p < 0.05.

| Condition (source-aware, primary) | Vocab | AP50@900 | Δ vs full | classes better / worse | robust |
|---|---|---|---|---|---|
| full | 80 | 0.0282 | ref | – | – |
| **RemoteCLIP, k=8** | 8 | **0.0582** | **+0.030** | 35 / 16 | **yes** |
| **CLIP, k=20** | 20 | 0.0492 | +0.021 | 39 / 11 | **yes** |
| **CLIP, k=8** | 8 | 0.0461 | +0.018 | 31 / 13 | **yes** |
| CLIP, k=3 | 3 | 0.0686 | +0.040 | 26 / 20 (p = 0.46) | no: gain in few classes |
| RemoteCLIP, k=3 | 3 | 0.0643 | +0.036 | 23 / 25 | no |
| CLIP / RemoteCLIP, k=40 | 40 | 0.040 / 0.040 | +0.012 / +0.011 | 38/5, 34/8 | no (basic CI includes 0) |
| frequency prior, k=3–40 | – | 0.004–0.017 | −0.011 to −0.024 | mostly worse | no (the basic CI just includes 0; robust under standard at k=3/8) |
| random, k=3–20 | – | 0.002–0.011 | −0.018 to −0.026 | mostly worse | no |
| oracle (GT classes) | 2.2 | 0.1404 | +0.112 | 66 / 3 | yes |

Under the **standard** protocol, every CLIP/RemoteCLIP point estimate is positive (+0.001 to +0.027), but **no CLIP-based gain is robust**: all their basic CIs include 0. Frequency and random are negative. The frequency prior at k = 3 and 8, and random at k = 3, are **robustly worse** than `full`; random at k = 8 is not (sign-test p = 0.154).

### Findings
1. **H2 is supported, moderately, under the primary protocol.**
   - Deployable global selectors at **medium budgets (k = 8–20)** improve source-aware AP50@900 over the full vocabulary by +0.018 to +0.030, i.e. ×1.6–2.1. This is robust to all three checks.
   - The best robust setting, **RemoteCLIP k = 8, recovers ~27% of the oracle headroom** (0.030 of 0.112).
   - Under the standard protocol the evidence is not robust.
   - **This satisfies the M0 gate, so M6/M7 may proceed.**
2. **Small budgets give the largest average gain but are fragile.** At k = 3 the gain comes from a few classes (DOTA/DIOR large objects; e.g. per-source source-aware AP50 for DOTA is 0.390 vs 0.113 for full with CLIP k = 3), while about as many classes get worse. The robust optimum at a *moderate* budget fits the project's hypothesis: a trade-off between keeping plausible classes (recall) and removing confusers. That motivates uncertainty-aware retention (M6) and an adaptive budget (M7).
3. **Small objects are not helped, and at small k they are hurt.**
   - Small-object R50 (standard): full 0.106 → CLIP k = 3 **0.001**, RemoteCLIP k = 3 0.071, RemoteCLIP k = 8 0.102.
   - FAIR (91% small objects) gains nothing: source-aware per-source AP50 0.011 (full) vs 0.004–0.017 across the CLIP/RemoteCLIP settings (computed from `per_class` in `metrics.json`).
   - **The gains come from medium and large objects.** This confirms the step-1 diagnosis that global embeddings miss small-object classes. Local evidence (M5) is the priority.
4. **Image-level vocabulary recall is a misleading proxy for detection AP.** The frequency prior has the *highest* image-level vocabulary recall at k ≥ 2, yet detects *worse* than `full`. It always proposes the same few classes, so most classes are never predicted and class-averaged AP collapses. CLIP's class-wise ranking mAP (0.14–0.18 vs 0.05) ordered the methods correctly. Selector quality should be reported with class-wise metrics too.
5. **Efficiency (H6, provisional).**
   - GPU selector time is ~8 ms/image for either CLIP (`outputs/selection_gpu/dev/comparison_vs_committed.json`).
   - Detector median latency: 467 ms (full) vs 363–372 ms (k = 3–8) and 379–387 ms (k = 20).
   - End-to-end, about **380 ms vs 467 ms (≈ −19%) at k = 8**.
   - **Caveat:** the GPU was not idle at the start (2,097 MB in use, 41% utilisation); full-vocabulary latency was 418 ms in E0. Conditions are interleaved per image, so the comparison is fair *within* the run, but absolute latencies must be re-measured on an idle GPU before the paper.

## Limitations
- The frequency prior is evaluated in-sample on dev.
- CLIP templates are fixed and untuned; class-name ambiguity across sources is not handled; images are center-cropped to 224 px by the CLIP preprocessing (PP images are non-square).
- Baseline E (caption-based) is not done yet.
- CPU timings are not latency results.

## Exit criteria (M4)
| Criterion | Status |
|---|---|
| Baseline comparison table | ✅ selection-level and detection-level (A–D); E deferred |
| Vocabulary-size trade-off curves | ✅ recall vs k; AP at k ∈ {3, 8, 20, 40} |
| Baseline E (caption-based) | ⏳ deferred |
