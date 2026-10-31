# M8/M9 — Experiment Framework & Metrics (2026-10-31)

## M8 — controlled experiment framework (status ✅)
- **One runner:** `scripts/run_detection.py` takes a YAML config: split, sample, detector checkpoint, precision, device, top-k, seed, and conditions. Condition types:
  - `full`
  - oracle diagnostics
  - `sel:<selector>:k<k>` (top-k from precomputed scores)
  - `set:<name>` (variable-size boolean masks)
- **Provenance:** every run has a config-hash run id and records git commit and dirty flag, host, torch/CUDA, checkpoint lock, GPU state at start and end, and the image sample.
- **Run behaviour:** runs are resumable, per-image records are written incrementally, and failures are logged and kept in the denominator.
- **One evaluator:** `scripts/evaluate_run.py` scores both protocols, with bootstraps against any reference (`--reference`, `--tag`).
- **Generated tables:** `scripts/report_run.py` and `scripts/compare_conditions.py`.
- **Registry:** `scripts/build_registry.py` writes `outputs/metrics/registry.csv` and `reports/generated/registry.md`. That is one row per (run, condition) across E0, M4, M5 and M6 (51 rows), mapped to the master-prompt experiment IDs:

| ID | Experiment | Status |
|---|---|---|
| E1 | full | ✅ |
| E2 | frequency prior | ✅ |
| E3 | random | ✅ |
| E4 | global CLIP / RemoteCLIP | ✅ |
| E5 | caption-based (VocAda-style) | **not yet done** |
| E6 | global + local | ✅ |
| E7 | retention | ✅ |
| E8 | adaptive budget | ✅ |

## M9 — evaluation and metrics (status ✅ for the dev phase)
| Metric family | Implemented and validated |
|---|---|
| Detection | AP, AP50, AP75 at maxDets 100/300/900; AR; AP50/R50 per COCO size; per-class AP; **P/R/F1 at fixed operating points (new)**. Two protocols (standard, source-aware) |
| Vocabulary | selected size (mean, sd, min–max), reduction, image-level vocabulary recall, full coverage, irrelevant-class rejection, pair recall by object size and class frequency, class-wise ranking mAP, macro pair recall |
| Efficiency | per-stage latency (preprocess / forward / postprocess), selector time (GPU), peak GPU memory, text tokens, GPU snapshot at start and end |
| Statistics | paired image bootstrap (percentile + basic CI, bias diagnostics), per-class sign test, D26 robustness rule |
| Validation | known-answer tests: GT → AP = 1; cross-source FP; maxDets cap; fast AP50 = COCOeval to 1e-9; P/R/F1 known answer. 56 tests in total |

Re-scoring all four detection runs with the extended evaluator changed **no AP value** (max |Δ| = 0.0). The new metrics were added without side effects.

### Operating points (P/R/F1 at IoU 0.5) — finding
Source: `reports/generated/m6_retention_det_tables.md`, "Operating points".
- **The pre-declared thresholds 0.25 and 0.35 (GDINO's common box threshold, as used by arXiv 2601.22164) are far too high for GDINO-T on aerial imagery.** Recall collapses for every condition. Even the **oracle** vocabulary has R@0.25 = 0.059 and R@0.35 = 0.022 (P = 0.30–0.33).
- This reproduces the prior study's "high precision, catastrophic recall" observation; their oracle recall was 3.9% at the same thresholds. It also confirms that a **threshold-free metric (AP) is the right headline**.
- The best-F1 thresholds lie at 0.11–0.17 for every condition. That is a diagnostic only, because they are chosen on the evaluated images. At best F1:

| condition | standard | source-aware |
|---|---|---|
| full | 0.014 | 0.039 |
| RemoteCLIP top-8 | 0.032 | 0.055 |
| threshold m≈8 (headline) | 0.032 | 0.050 |
| top_p m≈20 | 0.031 | 0.055 |
| oracle | 0.196 | 0.196 |

  Selection roughly doubles the best attainable F1 under standard scoring. Under source-aware scoring the gain is smaller, and the headline method is not the best on F1.

**D35 (fixed now, before test):** for the final test evaluation, F1 is reported at the score threshold that maximised F1 for the headline method on these dev images (**0.14 standard, 0.12 source-aware**), applied unchanged to test. The fixed 0.25/0.35 points are also reported, for comparability with prior work.

## Decision point (D31): detector-side small-object test
**Evidence so far (dev):**
- **E0:** even with the oracle vocabulary, small-object AP50 is 0.100, and 18/75 classes are near 0.
- **M5:** crop evidence raises small-object *selection* recall (+0.083) and small-object detection recall, but small-object AP falls.
- **M6:** retention raises small-object R50 (0.102 → 0.151) but not AP.
- **The headline adaptive method gains mainly on large objects.**

Conclusion: **small-object performance is limited by the detector at 800/1333 px, not by the vocabulary.** The only way to test that directly is a detector-side change (tiling or higher-resolution input), applied identically to all conditions.

**Decision (user, 2026-10-31; D36):** run one small tiling diagnostic. The config is `configs/experiments/m9_tiling_diagnostic.yaml`: full / oracle / headline selector, each with and without 2×2 tiles plus the full image. A results section will be added after the GPU run.
