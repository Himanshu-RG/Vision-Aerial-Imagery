# M4 — Master Vocabulary & Simple Baselines (in progress, 2026-10-25)

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
5. **Cost:** ~50 ms per image for either CLIP on the laptop CPU. GPU timing for H6 is pending on the RTX 5060. E0 implies a budget of ≲ 85 ms per image for a net latency win.

## Step 2 — detection with selected vocabularies (pending, RTX 5060)
- Config `configs/experiments/m4_selectors_detection.yaml`: the same 200-image dev sample and detector settings as E0.
- Conditions: `full`, `oracle_gt`, plus top-k ∈ {3, 8, 20, 40} for frequency / CLIP / RemoteCLIP, plus random k ∈ {3, 8, 20}. That is 17 conditions, roughly 20 minutes on the GPU.
- Selector scores come from the committed `.npz` files, so both machines use identical vocabularies.
- CPU smoke run (2 images, all 17 conditions): all ok, with correct vocabulary sizes and token counts; the evaluator produced 16 comparisons. Not a result; deleted.
- **Decision rule (from M0):** if no deployable selector beats `full` (paired CI above 0) under the source-aware protocol, H2 fails for global selectors. We then report it, and M5 (global–local evidence) is the next test before the M6/M7 gate.

## Limitations
- The frequency prior is evaluated in-sample on dev.
- CLIP templates are fixed and untuned; class-name ambiguity across sources is not handled; images are center-cropped to 224 px by the CLIP preprocessing (PP images are non-square).
- Baseline E (caption-based) is not done yet.
- CPU timings are not latency results.

## Exit criteria (M4)
| Criterion | Status |
|---|---|
| Baseline comparison table | 🟡 selection-level ✅; detection-level ⏳ (step 2) |
| Vocabulary-size trade-off curves | 🟡 recall vs k ✅; AP vs k ⏳ |
| Baseline E (caption-based) | ⏳ deferred |
