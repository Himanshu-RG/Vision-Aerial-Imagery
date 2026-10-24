# M3 — Grounding DINO Baseline Pipeline (+ M9 evaluator validation) (2026-10-24)

## Objective
Build a trustworthy, reproducible reference pipeline (frozen Grounding DINO → predictions → evaluation) and validate the metric code on known answers **before** any reported number exists.

## Status
- Pipeline implemented and validated on the dev laptop (CPU, 3-image smoke test).
- **No experimental results yet.** E0 runs on the RTX 5060; it is the next step.

## Files
| Path | Purpose |
|---|---|
| `src/aerial_ovd/detector/gdino.py` | `GroundingDINO` wrapper (HF transformers 4.57.3): official caption format, positive map, class scoring, top-k selection, per-stage timing |
| `src/aerial_ovd/vocabulary/conditions.py` | E0 conditions: `full` (deployable), plus oracle diagnostics `oracle_gt`, `oracle_source` and the size-matched control `random_matched` |
| `src/aerial_ovd/evaluation/coco_eval.py` | `evaluate` (standard / source-aware; AP, AP50, AP75 at maxDets 100/300/900; AR; per-size; per-class); `per_image_matches`, `ap50_from_matches`, `paired_bootstrap` |
| `scripts/run_detection.py` | Config-driven runner: deterministic exact-size stratified sample, resume, incremental JSONL, failure/OOM logging, warm-up, GPU snapshot, provenance |
| `scripts/evaluate_run.py` | Both protocols, latency/memory summary, paired bootstrap vs `full`, fast-vs-COCOeval consistency check, writes `outputs/metrics/<run_id>/{metrics.json, run_config.json}` |
| `configs/experiments/e0_headroom.yaml` | E0 configuration |
| `configs/checkpoints.lock.json` | GDINO-T pinned at commit `a2bb814dd30d776dcf7e30523b00659f4f141c71`, sha256 of every file (e.g. `model.safetensors` `1a2412ef…42f3`) |
| `tests/test_coco_eval.py`, `tests/test_gdino_and_conditions.py` | 16 new tests (39 in total) |

## Inference protocol (recorded per run in `run_config.json`)
| Item | Value | Source |
|---|---|---|
| Checkpoint | HF `IDEA-Research/grounding-dino-tiny` @ `a2bb814d…` (Swin-T; 900 queries; `max_text_len` 256) | lock file |
| Image preprocessing | resize shortest edge 800, longest ≤ 1333, bilinear; ImageNet mean/std; batch 1 (no padding) | `preprocessor_config.json`; matches the official `test_ap_on_coco.py` |
| Caption | `"name1 . name2 . … ."`, lower-case, light names (D14); category-id order for every condition | official script |
| Class score | sigmoid(token logits) @ row-normalised positive map, i.e. the **mean** over the name's tokens (**D20**, official). `max` is available only as an ablation. *Supersedes D5's "max".* | official `PostProcessCocoGrounding` |
| Selection | top-**900** (query, class) pairs per image (**D21**); no NMS, no score threshold for AP | official uses 300; we use 900 because GT density reaches 668 boxes/image |
| Boxes | cxcywh (normalised) → xyxy pixels, clipped to the image | — |
| Precision | fp32 (fp16 is a separate later ablation) | config |
| Thresholds | none for AP. A P/R/F1 operating point will be chosen on dev in M9 | — |
| Latency | per (image, condition): preprocess + forward + postprocess, with CUDA sync. Excludes disk I/O (`load_ms`) and model load (`model_load_s`), which are recorded separately. 3 warm-up passes not recorded | runner |

## Evaluation (D15, D21) — validated
- **Protocols:**
  - `standard`: all categories.
  - `source_aware`: predictions of classes outside the image's source label space are removed before evaluation.
- **Metrics:**
  - AP, AP50 and AR at maxDets **100** (official-comparable) and **900** (headline, dense aerial scenes);
  - AP75;
  - AP50 and R50 per COCO size bin;
  - per-class AP50, AP and R50.
- **Bootstrap:** paired, image-level bootstrap of the AP50@900 difference vs. `full` (1,000 resamples, seed 0). It uses per-image matches from one COCOeval pass and re-accumulates them exactly as pycocotools does.

**Known-answer tests (all pass):**
- Ground truth given as predictions gives AP = AP50 = AR = 1.0 at every maxDets, under both protocols.
- A confident cross-source "vehicle" box on a FAIR-style image lowers that class's AP under `standard` but not under `source_aware`.
- Wrong boxes give AP50 = 0. An empty prediction list is handled (AP = 0).
- An image with no predictions stays in the denominator (AP50 halves in the 2-image case).
- maxDets = 100 caps recall at 100/150 in a 150-object image; maxDets = 900 recovers 1.0.
- The fast AP50 reproduces COCOeval exactly (to 1e-9) on 4 random cases. A bootstrap of identical conditions gives a difference and CI of exactly 0.
- `evaluate_run.py` re-checks fast vs. COCOeval AP50 on real data for every condition and protocol, and aborts on mismatch. It passed on the smoke run.

## Smoke test (dev laptop, CPU) — pipeline check only, NOT a result
`python scripts/run_detection.py --config configs/experiments/e0_headroom.yaml --device cpu --n-images 4` processed **3** images, because the sampler rounded. This bug is now fixed: the sampler returns exactly n, with a test.
- All 12 (image, condition) records: status ok, 900 detections each.
- Text tokens: 249 for the full vocabulary, 4–110 for the oracle conditions.
- Model load 7.3 s. CPU forward pass 3.9–6.7 s per image (not a latency result; D9).
- The scoring pipeline produced metrics, bootstrap intervals, and a passing consistency check.

With 3 images the numbers have no statistical meaning, so they are deliberately not reported or committed. The smoke metrics were deleted.

## E0 sample (frozen by config + seed)
200 dev images (stratified, exact size; sample seed 2026):
- FAIR 119, xView 41, DIOR 30, DOTA 8, PP 2;
- 75 of the 79 classes present in dev are covered;
- test images are never touched (checked by a test).

## Decisions
- **D20 (supersedes D5):** class score = mean over the name's tokens (official positive-map averaging). `max` is an ablation only.
- **D21:** keep the top 900 (query, class) pairs per image. The headline is AP50 at maxDets 900; maxDets 100 is also reported for comparability. Justification: GT density goes up to 668 boxes/image (mean 24.1), so a 100-detection cap would bound recall regardless of vocabulary.
- **D22:** E0 sample = 200 dev images, exact-size class-stratified, seed 2026. The latency definition is as in the protocol table above.

## Known limitations
- E0 runs at a single resolution (800/1333). 59% of objects are small, so tiling may be needed later; if so, it applies identically to all methods.
- P/R/F1 at an operating point is not implemented yet (M9).
- Bootstrap CIs cover AP50@900 only.
- Per-class AP on 200 images covers only the classes present in the sample.
- HF transformers re-implements Grounding DINO. Its numerical equivalence to the official repo is assumed from the shared checkpoint and protocol, not verified against the official code.

## Exit criteria
| Criterion | Status |
|---|---|
| Reproducible inference with recorded parameters | ✅ (pinned checkpoint + lock, config hash run id, provenance) |
| Validated metrics | ✅ known-answer tests + real-data consistency check |
| Saved predictions | ✅ JSONL per condition (not committed); metrics + run config committed |
| Smoke test before the full run | ✅ CPU (laptop); ⏳ GPU smoke on the RTX 5060 |
| Baseline numbers | ⏳ E0 on the RTX 5060 |

## Next
1. The RTX 5060 runs the GPU smoke test, then E0 (commands in CONTEXT §12 / the chat runbook).
2. Evaluate E0, then decide per the M0 gates: H1 (headroom) and H7 (label-space share).
3. Then M4 baselines.
