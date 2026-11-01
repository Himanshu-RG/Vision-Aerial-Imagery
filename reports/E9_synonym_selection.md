# E9 — Synonym-aware vocabulary selection (Option A of D41; complete 2026-11-01)

**Motivation (M10 finding 2):** 59% of the classes the headline method selects belong to another source's label space, often a near-synonym of the annotated class. A diagnostic restricted to the right source raised object coverage from 0.25 to 0.68. This experiment was **motivated by the dev failure analysis**; the test split is untouched.

## Method and protocol (D42, amended D42a before any detection)
- **Rule:** the headline rule (RemoteCLIP p_g ≥ t), then add every class whose class-name text similarity to a selected class is ≥ τ. Similarity comes from the RemoteCLIP text encoder with the selector's own templates. The rule uses class names only.
- **Tuning on the 530 calibration images only:**
  - The τ grid is fixed in advance.
  - For each τ, t is calibrated to a final mean size of 8.
  - τ is chosen by J = (macro pair recall + object-weighted coverage) / 2.
  - **D42a:** τ values that cannot reach size 8 are ineligible. Without this, a size-38 vocabulary had been chosen, breaking the matched-size comparison.
- **Chosen:** τ = 0.925, t = 0.0334. It links 41 class pairs.
- **Replacement rule (pre-declared):** E9 replaces D34 for test only if it beats D34 on AP50 under both protocols by point estimate, and the source-aware per-class sign test favours it with p < 0.05.

## Selection quality
Source: `reports/generated/E9_synonym_tables.md`.

| selection | mean size | macro pair recall | object-weighted coverage | out-of-source share |
|---|---|---|---|---|
| headline (D34), calibration images | 8.0 | 0.466 | 0.328 | – |
| E9 τ = 0.925, calibration images | 8.0 | 0.495 | 0.310 | – |
| headline (D34), held-out 200 | 7.8 | 0.498 | 0.260 | 0.59 |
| **E9 m≈8, held-out 200** | 7.6 | 0.453 | 0.218 | 0.58 |
| E9 with D34's t, held-out 200 | 9.6 | 0.550 | 0.281 | 0.59 |
| DIAGNOSTIC in-source m≈8, held-out 200 | 8.2 | 0.768 | 0.809 | 0 |

### Finding (selection level)
- **At matched size, synonym expansion does not help.** The calibration gain is small (J 0.397 → 0.403), and on held-out images both coverage measures drop.
- **Cause:** the text encoder does not separate synonyms from merely related classes. Warship ↔ yacht (0.964) and engineering ship ↔ warship (0.961) score higher than small car ↔ passenger car (0.920). So the expansion spends the budget on related ship types.
- The in-source diagnostic shows that the label-space problem is real and large, but fixing it needs to know which label space an image uses. Inferring the source dataset from the image would mean fitting the benchmark's construction rather than the task, so it is not pursued.

## Detection (RTX 5060, same 200 dev images)
**Run.** `e9_synonym_det-cd6a2ee199e9`: code `fc4653e` (clean), GPU idle at start (1,518 MB, 0%), 6 conditions, 0 failures. Pushed by Amit as `5765647`. The shared conditions reproduce earlier runs exactly: full 0.0146 / 0.0282, oracle 0.1404, headline 0.0550 / 0.0707.
Tables:
- `reports/generated/e9_synonym_det_tables.md` (vs full)
- `reports/generated/e9_synonym_det_tables_vs_headline.md` (vs headline)

| condition | size | AP50 std | AP50 src-aware | Δ vs headline std / src (classes better/worse, sign p) | best F1 std | R50 small std |
|---|---|---|---|---|---|---|
| headline (D34) | 7.8 | 0.0550 | 0.0707 | ref | 0.032 | 0.103 |
| **E9 m≈8** | 7.6 | 0.0516 | 0.0707 | −0.003 / +0.000 (17/24, p = 0.35; 18/17, p = 1) | 0.027 | 0.135 |
| E9 with D34's t | 9.6 | 0.0529 | 0.0695 | −0.002 / −0.001 (n.s.) | – | 0.154 |
| DIAGNOSTIC in-source m≈8 | 8.2 | 0.0760 | 0.0760 | +0.021 / +0.005 (41/12, p = 8e-5; 31/21, p = 0.21) | 0.066 | 0.248 |

### Decision (pre-declared rule, D42 → D43)
**E9 does not replace D34.** It is not better under either protocol (−0.003 standard, +0.000 source-aware) and the sign test is flat. **D34 stays frozen as the headline for the one-time test.**

### Findings
1. **Synonym expansion from class-name text similarity is a negative result.** The selector's own text encoder cannot tell synonyms from related classes, so at a matched budget it trades kept classes for neighbours. It is reported as such.
2. **Even a perfect fix to the label-space problem helps class-averaged AP only a little.**
   - The in-source diagnostic, which knows each image's label space, raises object coverage from 0.26 to 0.81.
   - Under standard scoring it gains +0.021 AP50 (sign test p = 8e-5).
   - Under source-aware scoring the gain is only +0.005 (n.s.).
   - Its gains are object-level: best F1 doubles (0.032 → 0.066), and small-object R50 rises from 0.103 to 0.248.
   - **So the cross-source synonym problem found in M10 mostly costs dense, frequent-class objects, which pooled metrics count and class-averaged AP barely weighs.** The detector remains the limit on those classes (M10 findings 3–4).
3. Neither the headline result nor the D35 operating points change.
