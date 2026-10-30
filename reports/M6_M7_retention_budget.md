# M6/M7 — Uncertainty-Aware Retention & Adaptive Budget (steps 1–2 done, 2026-10-31)

## Objective
- **M6:** stop the selector from dropping plausible classes when its evidence is ambiguous (H4).
- **M7:** let the vocabulary size vary per image instead of a fixed k (H5).

Both are built on the global RemoteCLIP selector (D30). Local evidence is used only as an uncertainty signal.

## Rules (`src/aerial_ovd/vocabulary/retention.py`, 5 unit tests)
Signals per image:
- **p_g:** RemoteCLIP global class probabilities (softmax of logit_scale × cosine).
- **p_l:** mean softmax over the 13 crops (2×2 + 3×3) from M5.

| rule | definition | role |
|---|---|---|
| `topk` | top-k by p_g | fixed-budget baseline (M4) |
| `retain` (M6) | top-⌈m/2⌉ by p_g, **plus** every class with p_l − p_g ≥ d (local evidence much stronger than global); at most 40 classes | uncertainty-aware retention via global–local disagreement (a signal the master prompt lists) |
| `threshold` (M6/M7) | keep classes with p_g ≥ t | calibrated / conformal-style; the size adapts per image |
| `top_p` (M7) | smallest set whose cumulative p_g ≥ τ | adaptive budget: the set grows when the probability mass is spread out (ambiguity) |

Every rule keeps at least 1 and at most 40 classes.

## Protocol (D32, fixed before results)
- **Calibration set:** the 530 dev images *not* in the 200-image detection sample.
- **Evaluation set:** the 200-image sample (E0/M4/M5).
- Each rule's single parameter is calibrated on the calibration set by bisection, to a **target mean size m ∈ {3, 8, 20}**. It is never tuned to a score.
- On the evaluation set, each rule is compared with fixed top-k **at the same achieved mean size** (top-k curve interpolated over k = 1…40), plus a paired image bootstrap vs top-k = m.
- **Conformal-style check:** threshold at the calibration quantile of the global probabilities of present classes, for target pair coverage 0.5 / 0.8 / 0.9.
- **Command:** `python scripts/m6_retention.py` (10 s). Tables: **`reports/generated/M6_retention_tables.md`**. Figure: `outputs/figures/m6_recall_vs_size.png`.

## Results (evaluation set, 200 images; from the generated tables)
Calibration transferred well: achieved mean sizes were 2.88–3.04, 7.77–8.02 and 19.78–20.26 for targets 3, 8 and 20.

| target | rule | mean size (sd) | image vocab recall | vs top-k at same size | small-pair recall | vs top-k | Δ vs top-k=m (95% CI) |
|---|---|---|---|---|---|---|---|
| 3 | topk | 3.00 (0) | 0.217 | – | 0.071 | – | ref |
| 3 | **retain** | 3.04 (0.9) | **0.266** | **+0.046** | **0.190** | **+0.117** | **+0.048 [+0.012, +0.084]** |
| 3 | threshold | 2.88 (1.4) | 0.219 | +0.006 | 0.059 | −0.010 | +0.002 (n.s.) |
| 3 | top_p | 2.96 (1.9) | 0.194 | −0.021 | 0.047 | −0.023 | −0.023 (n.s.) |
| 8 | topk | 8.00 (0) | 0.388 | – | 0.213 | – | ref |
| 8 | retain | 8.02 (2.0) | 0.417 | +0.029 | **0.312** | **+0.098** | +0.029 (n.s.) |
| 8 | threshold | 7.77 (3.8) | 0.412 | +0.031 | 0.186 | −0.019 | +0.024 (n.s.) |
| 8 | top_p | 7.93 (5.5) | 0.390 | +0.004 | 0.209 | −0.001 | +0.002 (n.s.) |
| 20 | topk | 20.00 (0) | 0.662 | – | 0.514 | – | ref |
| 20 | retain | 20.26 (5.0) | 0.625 | −0.041 | 0.502 | −0.016 | −0.037 (n.s.) |
| 20 | threshold | 19.78 (10.4) | 0.668 | +0.008 | 0.490 | −0.018 | +0.006 (n.s.) |
| 20 | top_p | 20.00 (11.7) | 0.672 | +0.010 | 0.526 | +0.012 | +0.010 (n.s.) |

Conformal-style coverage (target → achieved on the evaluation set; mean size):

| target | achieved | mean size |
|---|---|---|
| 0.50 | 0.484 | 13.3 |
| 0.80 | 0.756 | 27.3 |
| 0.90 | 0.811 | 31.5 (sizes capped at 40) |

## Findings (selection level)
1. **Retention via global–local disagreement (M6) helps at small budgets.**
   - At ~3 classes, image vocabulary recall rises by +0.048 (CI excludes 0), and small-object pair recall rises from 0.071 to 0.190.
   - At ~8 classes the small-object gain persists (+0.098), but overall recall is n.s. and class-averaged recall drops (−0.061): the added classes concentrate on frequent small-object classes.
   - At ~20 classes it is worse than top-k.
2. **Adaptive budgets (M7) do not beat a fixed top-k at matched mean size.** `threshold` and `top_p` stay within ±0.03 of top-k (n.s.), and `top_p` is worse at small sizes. RemoteCLIP's confidence is not informative enough to size the set per image. **H5 is not supported at the selection level.**
3. **Calibrated coverage roughly holds but is expensive.** Coverage targets are met within 0.02–0.09 (slightly under, plausibly because of the 40-class cap and a 200-image evaluation set). Keeping 80% of present classes needs ~27 classes per image, which shows how weak the per-image relevance signal still is.
4. **Expectation for detection (H4):** M5 showed that extra small-object classes did not turn into AP. The retain rule's gains are mostly small-object recall, so a detection gain is uncertain. That is what step 2 tests.

## Step 2 — detection (RTX 5060)
**Run.**
- `m6_retention_det-8febbe4deb4a`: code `0f8e1b4` (clean), the same 200 images, 14 conditions, **0 failures**. Pushed by Amit as `57384b4`.
- The baselines (`full`, `oracle_gt`, RemoteCLIP top-k) reproduce E0/M4/M5 exactly.
- **GPU not idle:** 2,341 MB in use, 43% utilisation at the start (the display runs on the RTX 5060). Latencies (`full` median 549 ms vs 418–467 ms in earlier runs) are therefore inflated. Within-run comparisons remain fair because conditions are interleaved per image.

**Tables (all generated by `scripts/report_run.py`):**
- vs `full`: `reports/generated/m6_retention_det_tables.md`
- vs top-k at matched size: `reports/generated/m6_retention_det_tables_vs_k3.md`, `_vs_k8.md` and `_vs_k20.md`

These include the full D26 robustness checks; the bootstraps were run on the RTX 5060 with `--reference sel:remoteclip_b32:kK`.

### Against the full vocabulary (D26: percentile CI + basic CI + per-class sign test)
| condition | mean size | AP50 standard | Δ (robust?) | AP50 source-aware | Δ (robust?) |
|---|---|---|---|---|---|
| full | 80 | 0.0146 | ref | 0.0282 | ref |
| RemoteCLIP top-8 (M4) | 8 | 0.0349 | +0.020 (no) | 0.0582 | +0.030 (**yes**) |
| **threshold, m≈8** | 7.8 | **0.0550** | **+0.040 (yes)** | **0.0707** | **+0.042 (yes)** |
| top_p, m≈8 | 7.9 | 0.0413 | +0.027 (no) | 0.0651 | +0.037 (**yes**) |
| **threshold, m≈20** | 19.8 | 0.0469 | +0.032 (**yes**) | 0.0686 | +0.040 (**yes**) |
| **top_p, m≈20** | 20.0 | 0.0494 | +0.035 (**yes**) | 0.0673 | +0.039 (**yes**) |
| retain, m≈3 / 8 / 20 | 3 / 8 / 20 | 0.037 / 0.033 / 0.019 | +0.022 / +0.018 / +0.004 (no) | 0.061 / 0.046 / 0.042 | (no) |
| threshold / top_p, m≈3 | 3 | 0.044 / 0.040 | (no) | 0.060 / 0.059 | (no) |

### Against fixed top-k at the same mean size
| condition | standard Δ vs top-k | source-aware Δ vs top-k | robust? |
|---|---|---|---|
| retain m≈3 / 8 / 20 | −0.005 / −0.002 / +0.001 | −0.003 / −0.012 / +0.002 | no |
| threshold m≈3 | +0.003 | −0.005 | no |
| threshold m≈8 | **+0.020** (sign test 33/11, p = 0.0013; basic CI includes 0) | +0.013 | no (2 of 3 checks under standard) |
| threshold m≈20 | **+0.029** (both CIs exclude 0; sign p = 0.079) | **+0.028** (both CIs exclude 0; sign p = 0.38) | no (2 of 3 checks) |
| top_p m≈8 | +0.006 | +0.007 | no |
| top_p m≈20 | **+0.032** | **+0.027** | **yes (standard)**; source-aware 2 of 3 |

Per-size (standard), RemoteCLIP top-8 → threshold m≈8:
- large-object AP50 0.088 → **0.128**
- medium AP50 0.020 → 0.022
- small AP50 0.035 → 0.036
- R50 small is unchanged (0.102 → 0.103)

### Findings (detection level)
1. **H5 (adaptive budget beats fixed top-k at matched size) is NOT established by the pre-declared criterion.** That criterion is source-aware and requires all three D26 checks; the best adaptive rules pass only 2 of 3. What *is* robust:
   - The calibrated-threshold rule at a mean of ~8 classes is the **best deployable selector so far**: AP50 0.0550 standard (×3.8 vs full) and 0.0707 source-aware (×2.5).
   - It is the **first selector whose gain over the full vocabulary is robust under the standard COCO protocol as well**.
   - Against fixed top-k at the same mean size, adaptive rules gain +0.020 to +0.032 at m ≈ 8–20. That is fully robust only for top_p m≈20 under the *standard* protocol; the rest pass 2 of 3 checks. These gains are **suggestive, not established**, and must be re-tested on the test split.
2. **Why adaptive sizes help even though selection-level recall did not improve:**
   - A per-image threshold gives *few* classes to images where RemoteCLIP is confident (less confusion) and *more* to ambiguous images.
   - Image-level vocabulary recall does not capture this redistribution; class-averaged AP does.
   - The gain is concentrated in **large objects** (AP50 0.088 → 0.128).
   - This is a second instance of D27: selection-level proxies can mislead in either direction.
3. **H4 (uncertainty-aware retention via global–local disagreement) is not supported at the detection level.** The retain rule raises small-object R50 (k≈8: 0.102 → 0.151) but not AP, at any size, which repeats the M5 pattern. The small-object bottleneck is the detector (risk 33; to be decided at M9, D31).
4. **Small budgets (m≈3) gain nothing from adaptivity.** All rules are within ±0.005 of top-3.
5. **Latency:** at a mean of ~8 classes, adaptive rules cost the same as top-8 (437 ms here vs 549 ms full, −20%; selector ~8 ms). The absolute numbers are inflated by the non-idle GPU.

## Caveats
- **Test-time protocol fixed now (D34):** the headline rule is used on test with the threshold calibrated on the 530 dev calibration images (t = 0.0237, from `outputs/selection/dev/m6_threshold_m8.npz` meta). It is not re-calibrated.
- **Selection among conditions.** The threshold m≈8 rule is the best of 9 rule × size conditions on the same 200 images. Its parameter was set on other images, but *choosing it as the headline* uses these 200. The claim must be confirmed once on the held-out test split; the rule and its m are fixed before that evaluation.
- Only one parameter per rule, and no rule was tuned on the evaluation images. But the rule *family* and the k_base = ⌈m/2⌉ choice were designed by us; this is reported.
- Evaluation set: 200 images. Selection-level CIs are percentile bootstraps of per-image means (appropriate; D26 scope note).

## Exit criteria (M6/M7)
| Criterion (master prompt) | Status |
|---|---|
| Retention rule + uncertainty signal defined, with min/max budget and calibration protocol | ✅ (`retention.py`, D32) |
| Ablation: global-only vs global-local vs with uncertainty retention | ✅ (M5 + M6, selection and detection) |
| Evidence on recall / small objects / latency at matched size | ✅ (H4 not supported at detection level) |
| Adaptive vs fixed budgets at several operating points, with size distributions | ✅ (H5 not established vs top-k; adaptive threshold robustly better than full under both protocols) |

