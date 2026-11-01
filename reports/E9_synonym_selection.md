# E9 — Synonym-aware vocabulary selection (Option A of D41; step 1 done 2026-11-01, detection pending)

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

## Detection (pending)
Config: `configs/experiments/e9_synonym_detection.yaml`, with 6 conditions: full, oracle, headline, E9 m≈8, E9 with D34's t, and the in-source diagnostic. The CPU smoke run is OK.
