# Reproducing every result

This guide goes from a fresh clone to every number in the reports. Decisions (D-numbers) are in [CONTEXT.md](CONTEXT.md) §8.

## 0. Quick check (dev laptop, ~1 minute, no GPU)
```bash
pytest                                   # 67 tests
python scripts/verify_artifacts.py       # expect "ALL REPRODUCED"
```
`verify_artifacts.py` needs a clean `reports/generated/` and `outputs/metrics/`. It checks three things:
1. **Frozen split.** The dev/test ID lists match the D18 sha256, with no overlap.
2. **Frozen methods.** The headline method and both secondaries reproduce every committed mask (dev, test, NWPU) from the committed RemoteCLIP scores.
3. **Tables.** It re-runs 25 table generators from committed metrics and confirms no file changes.

## 1. Environment (both machines)
| | Dev laptop | Experiment machine (reference, D13) |
|---|---|---|
| Use | selection on CPU, analysis, tables, tests | all detection runs and all timings (D9) |
| torch | 2.7.1 CPU wheel | 2.7.1 + cu128 (RTX 5060, Blackwell) |
| Lock file | `requirements-lock-LAPTOP-8N8B1VIP.txt` | `requirements-lock-LAPTOP-5CPRTI27.txt` |

Install as in [README.md](README.md#install). Then:
```bash
python scripts/env_audit.py --write-lock
python scripts/fetch_checkpoints.py      # GDINO-T, RemoteCLIP, OpenAI CLIP: pinned commits, sha256 lock
```

## 2. Data (M2, D14–D18)
```bash
python scripts/prepare_lae80c.py --download              # clean GT, manifest, frozen split (refuses to re-split)
python scripts/fetch_lae80c_images.py --split all --delete-zip   # GPU machine; dev laptop: --split dev
python scripts/check_prompt_length.py --classes configs/lae80c_classes_light.txt   # 249 / 256 tokens
python scripts/prepare_nwpu.py                           # M11 external set (614 images, D45)
```
Check `<store>/datasets/LAE-80C-clean/*.json` against the sha256 values in `data/manifests/lae80c_prepare_report.json`.

## 3. Selection (dev laptop, CPU; outputs committed under `outputs/selection/`, D24)
| Step | Command | Output |
|---|---|---|
| M4 selector scores | `python scripts/run_selection.py --split dev --device cpu` | `outputs/selection/dev/{frequency,random_s*,clip_openai_b32,remoteclip_b32}.npz` |
| M4 selection quality | `python scripts/evaluate_selection.py --split dev` | `reports/generated/M4_selection_tables.md` |
| M5 crop views (not committed) | `python scripts/run_views.py --split dev --model remoteclip_b32` (and `clip_openai_b32`) | `<repo>/outputs/selection/dev/views_*.npz` (gitignored) |
| M5 global + local | `python scripts/m5_global_local.py --split dev` | `*_gl.npz`, `*_local.npz`, `M5_selection_tables.md` |
| M6/M7 rules + calibration | `python scripts/m6_retention.py` | `m6_*.npz` masks, `M6_retention_tables.md` |
| E5 captions (GPU or CPU) | `python scripts/run_captions.py --split dev`, then `python scripts/e5_caption_selector.py` | `captions_florence2_base.json`, caption/TagCLIP-style npz, `E5_selection_tables.md` |
| E9 synonym selector | `python scripts/e9_synonym_selector.py` | `e9_*.npz`, `E9_synonym_tables.md` |
| NWPU (frozen) | `python scripts/run_selection.py --split nwpu --device cpu --selectors frequency clip_openai_b32 remoteclip_b32`, then `python scripts/make_frozen_headline.py --split nwpu` | `outputs/selection/nwpu/` |
| Test (once, D47) | same two commands with `--split test` | `outputs/selection/test/` |

## 4. Detection runs (RTX 5060; raw predictions stay on that machine)
For each config, run these three commands:
```bash
python scripts/run_detection.py --config configs/experiments/<config>.yaml   # resumable; failures stay in the denominator
python scripts/evaluate_run.py outputs/predictions/<run_id> [--reference <cond> --tag <tag>] [--thresholds ...]
python scripts/report_run.py outputs/metrics/<run_id>[_vs_<tag>]
```
| Result | Config | Run id (committed metrics) | Extra analyses |
|---|---|---|---|
| E0 headroom | `e0_headroom.yaml` | `e0_headroom-e8585910cbfb` | `report_e0.py` |
| M4 baselines | `m4_selectors_detection.yaml` | `m4_selectors_det-16d38bd056f8` | GPU selector timing: `run_selection.py --device cuda --out-dir outputs/selection_gpu/dev`, `compare_selection.py` |
| M5 global + local | `m5_global_local_detection.yaml` | `m5_global_local_det-aab45f29ca9b` | `run_views.py --device cuda --grids 2 3 --out-dir outputs/selection_gpu/dev`, `compare_views.py` |
| M6/M7 (headline chosen, D34) | `m6_retention_detection.yaml` | `m6_retention_det-8febbe4deb4a` | `--reference sel:remoteclip_b32:k{3,8,20} --tag k{3,8,20}` |
| M9 tiling diagnostic | `m9_tiling_diagnostic.yaml` | `m9_tiling_diag-31ec5d53a242` | `--reference full@tile2 --tag tile_full` |
| E5 caption / TagCLIP-style | `e5_caption_tagclip_detection.yaml` | `e5_caption_tagclip_det-69e357c3b1f4` | `--reference sel:remoteclip_b32:k8 --tag rc8`; M10: `m10_failure_analysis.py`, `m10_vocab_coverage.py` |
| E9 synonym selector | `e9_synonym_detection.yaml` | `e9_synonym_det-cd6a2ee199e9` | `--reference set:m6_threshold_m8 --tag headline` |
| M11 NWPU VHR-10 | `m11_nwpu_detection.yaml` | `m11_nwpu_det-58d87bbcbac5` | `--reference set:headline_d34 --tag headline`; `m11_nwpu_subgroups.py`, `m11_nwpu_selection.py` |
| **Final test (D47)** | `final_test.yaml` | `final_test-75ab843910a3` | `--reference set:headline_d34 --tag headline --thresholds 0.12 0.14 0.25 0.35`; `m11_per_source.py` |
| M12 latency (D49) | – | `outputs/metrics/m12_latency/<host>.json` | `python scripts/m12_latency.py` (idle GPU required) |

Cross-run analyses (dev laptop, from committed metrics):
- `python scripts/m11_per_source.py outputs/metrics/<run_id>`: per source and seen/unseen groups.
- `python scripts/build_registry.py`: one row per (run, condition) in `outputs/metrics/registry.csv`.

## 5. What is not in git, and why
| Item | Where | Regenerate with |
|---|---|---|
| Images, cleaned GT | `<store>/datasets/` | §2 (pinned downloads, sha256-checked) |
| Checkpoints | `<store>/checkpoints/` | `fetch_checkpoints.py` (sha256 lock) |
| Raw predictions | `outputs/predictions/` on the RTX 5060 | §4 (same config gives the same run id) |
| M5 crop views | `outputs/selection/<split>/views_*.npz` | `run_views.py` |

LAE-80C is MIT-licensed. The NWPU VHR-10 images are for research use only and are not redistributed.

## 6. Known reproducibility limits
- **CPU vs GPU numerics.** GPU numerics can differ from CPU by ~1e-6. That is why selection is computed once on the CPU and committed (D24); the GPU re-computation matched (M4, M5).
- **Latency.** Earlier runs had a non-idle GPU. M12 re-measures with an idle check (D49).
- **Pytest record.** Pytest in the RTX 5060 reference environment has passed but was never recorded line by line.
