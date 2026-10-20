# M0 — Gap Statement, Hypotheses, Novelty Risk, Decision (2026-10-20)

## 1. Gap statement
Two findings frame the gap:
- On aerial imagery, Grounding DINO's zero-shot performance collapses when it is prompted with a large overlapping vocabulary. With an *oracle* per-image vocabulary it improves by an order of magnitude: F1 goes from 0.5% to 7.4% on LAE-80C [Tsourveloudis, arXiv 2601.22164].
- Test-time vocabulary adaptation has been shown to help OV detectors on natural images [VocAda, ICIP 2025], but it was not evaluated on Grounding DINO or on aerial imagery. Its captioner+LLM pipeline also costs 5–11 s and 17–31 GB per image.

**What no verified work has done** is test whether a *deployable, inference-time* selector, one that uses no ground truth, recovers that oracle headroom for a frozen Grounding DINO on aerial imagery. This includes:
- standard COCO metrics;
- breakdowns for small, rare, and visually similar categories;
- the 256-token prompt limit;
- honest end-to-end latency.

## 2. Hypotheses (falsifiable; tested on the dev split unless stated otherwise)
| ID | Hypothesis | Falsified if |
|---|---|---|
| H0 | The 80 LAE-80C class names exceed GDINO's 256-token limit, so a single-prompt "full vocabulary" is silently truncated | Tokenized length of the 80-class prompt ≤ 256 |
| H1 | Oracle per-image vocabulary > full vocabulary (chunked, no truncation) in mAP@50 | Image-bootstrap 95% CI of the difference includes 0 |
| H2 | ≥1 deployable global selector at a matched mean vocabulary size beats the full vocabulary in mAP@50 | No budget k achieves a CI above 0 |
| H3 | Global + local evidence > global-only in vocabulary recall for small-object classes at a matched mean vocabulary size | CI includes 0 |
| H4 | Uncertainty-aware retention > no retention in detection recall at a matched mean vocabulary size | CI includes 0 |
| H5 | Adaptive budget lies above the fixed-k Pareto curve (mAP vs. mean vocabulary size) | Adaptive points lie on or below the fixed-k curve |
| H6 | End-to-end latency (selector + detector) < full-vocabulary latency on the RTX 5060 | Measured median latency ≥ full |

### H0 result (2026-10-20): FALSIFIED for raw class names
- **Command:** `python scripts/check_prompt_length.py --classes configs/lae80c_classes_raw.txt` (tokenizer `IDEA-Research/grounding-dino-tiny@a2bb814dd3`; `max_text_len` = 256 in its config.json).
- **Result:** the 80 raw LAE-80C names in the format `"name1. name2. … name80."` (lower-cased) use **251 tokens** including [CLS]/[SEP]. That is ≤ 256, so a single full-vocabulary prompt is **not truncated**.
- Tokens per class (with period): min 2, max 8 (`Front loader/Bulldozer`), mean 3.11.
- **Caveat:** the margin is only 5 tokens. Any per-class prefix (e.g. "aerial view of"), synonym expansion, or longer naming policy **will** exceed 256. The token count must therefore be re-checked whenever the naming policy changes (M2), and chunking (D6) stays available for such variants. Whether the prior study's prompt variants were truncated is an open question that we have **not** verified.

## 3. Novelty-risk assessment
| Component | Risk | Reason | How we position it |
|---|---|---|---|
| Vocabulary pruning itself | **High** | VocAda | Not claimed. Cited as the foundation |
| Global CLIP selection | **High** | VocAda-CLIP; standard CLIP tagging | Baseline D |
| Global–local relevance | **Medium-high** | TagCLIP (local-to-global tagging, natural images) | Claim only an aerial-specific, detector-coupled design. Include a TagCLIP-style baseline |
| Uncertainty-aware retention | **Medium** | Conformal set prediction exists in general ML | Candidate principled rule: calibrate the selector on dev to a target *vocabulary recall* (coverage); report achieved coverage on test |
| Adaptive budget | **High** standalone | Adaptive top-k is standard | A consequence of the uncertainty/coverage rule, not a separate claim |
| Empirical study (oracle gap with COCO mAP, chunking confound, small/rare analysis, latency) | **Low-medium** | 2601.22164 covers part of it with P/R/F1 | Main defensible contribution together with the selector |

**Defensible contribution (if results support it):** the first systematic evaluation of deployable vocabulary selection for frozen Grounding DINO on aerial imagery, together with a recall-calibrated global–local selector, quantified against the oracle upper bound under matched budgets and honest latency.

**Venue realism:** with GDINO-T, a frozen detector, and one main benchmark, a CVPR/ICCV/ECCV main-track acceptance is **unlikely** unless effects are large and consistent across domains. Realistic targets are an EarthVision-type workshop, IGARSS, IEEE GRSL, or JSTARS. Acceptance is not guaranteed at any venue.

## 4. Decision: components justified for implementation
1. **Implement:** baselines A–E (M4), the oracle diagnostic, and the global–local selector (M5).
2. **Gate:** implement M6 (uncertainty/coverage retention) and M7 (adaptive budget) **only if** M4/M5 show a selector beating the full vocabulary on dev (H2 holds).
3. **If H1 fails** (no oracle headroom under COCO mAP): stop method work and pivot to an analysis paper explaining why the prior study's oracle gain does not carry over (e.g., truncation, metric choice).
4. **If H1 holds but H2 fails:** report the negative result, then analyse where selection errors occur (vocabulary recall vs. detection) before deciding on M5.
