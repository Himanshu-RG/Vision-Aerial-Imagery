# M6/M7 — Uncertainty-Aware Retention & Adaptive Budget (step 1 done, 2026-10-30; detection pending)

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

## Step 2 — detection (pending, RTX 5060)
- Config: `configs/experiments/m6_retention_detection.yaml`.
- 14 conditions: `full`, `oracle_gt`; RemoteCLIP top-k at 3/8/20; `set:m6_{retain,threshold,top_p}_m{3,8,20}`.
- CPU smoke run (1 image): all 14 ok, with variable sizes as intended.
- **Decision:** H4/H5 hold at the detection level only if a rule beats top-k at the same mean size under D26 (source-aware).
  - The full D26 check needs the bootstrap against each top-k reference, which needs the predictions. So the RTX 5060 runbook includes `evaluate_run.py --reference sel:remoteclip_b32:k{3,8,20} --tag k{3,8,20}`.
  - `scripts/compare_conditions.py` adds per-source breakdowns.
- **Provenance of the masks:** the masks come from the crop views computed before the M5 code commit. Amit's clean-commit GPU recomputation reproduced the same global and crop scores (max |diff| ≤ 1.2×10⁻⁵; identical M5 top-k sets), so the masks are not regenerated.

## Caveats
- Only one parameter per rule, and no rule was tuned on the evaluation images. But the rule *family* and the k_base = ⌈m/2⌉ choice were designed by us; this is reported.
- Evaluation set: 200 images. Selection-level CIs are percentile bootstraps of per-image means (appropriate; D26 scope note).
