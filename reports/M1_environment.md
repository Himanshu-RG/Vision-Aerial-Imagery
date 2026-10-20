# M1 — Repository & Compute Audit (2026-10-20)

## Objective
Set up a reproducible, resource-aware environment and project skeleton without destroying existing work.

## Pre-existing state
The workspace `Sem07/vision` had no code and no git repo. It held two PDFs:
- `Brief_Project_Description_Adaptive_Vocabulary_Aerial_OVD.pdf`: the project brief.
- `01.pdf`: an unrelated KnowScene paper.

Both PDFs are left untouched and are gitignored (`*.pdf`), so the copyrighted paper is never committed.

## Files created
| Path | Purpose |
|---|---|
| `CONTEXT.md` | Project ground truth (see its §12 "How to resume") |
| `.gitignore`, `pyproject.toml`, `README.md` | Repo hygiene, package metadata, install guide |
| `requirements.txt`, `requirements-gpu.txt` | Pinned stack (CPU / cu128) |
| `requirements-lock-LAPTOP-8N8B1VIP.txt` | `pip freeze` of the dev-laptop venv (51 packages after the 2026-10-21 alignment) |
| `configs/paths.example.yaml` | Template for the per-machine store path (`paths.local.yaml` is gitignored) |
| `configs/checkpoints.yaml` | Checkpoint list: GDINO-T, RemoteCLIP ViT-B/32, OpenAI CLIP ViT-B/32 |
| `configs/lae80c_classes_raw.txt` | The 80 LAE-80C class names, verbatim from `LAE-80C-benchmark.txt` on HF |
| `src/aerial_ovd/utils/{seed,logging,config,run_id,paths}.py` | Seeding, JSONL logging, strict pydantic configs, config-hash run IDs, store resolution |
| `src/aerial_ovd/{data,detector,vocabulary,evaluation,experiments}/` | Empty subpackages for later modules |
| `scripts/env_audit.py` | Writes `outputs/env/<hostname>.json` |
| `scripts/fetch_checkpoints.py` | Resolves Hub commits and sizes, checks disk, downloads, writes sha256 lock |
| `scripts/check_prompt_length.py` | H0 token-length check |
| `tests/*.py` | 16 unit tests |
| `reports/M0_*.md`, `reports/M1_environment.md` | Module reports |

## Commands run (dev laptop)
```bash
python -m venv D:/aerial-ovd-store/venv          # TEMP and pip cache redirected to D: (C: has 3.7 GB free)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.57.3 open_clip_torch pycocotools pyyaml pydantic pandas matplotlib pytest psutil huggingface_hub safetensors
pip install -e .
pytest                                            # 16 passed in 7.38 s
python scripts/env_audit.py                       # outputs/env/LAPTOP-8N8B1VIP.json
python scripts/fetch_checkpoints.py --dry-run
python scripts/check_prompt_length.py --classes configs/lae80c_classes_raw.txt
```

## Actual results
**Dev laptop (`LAPTOP-8N8B1VIP`):**
- Windows 10.0.26300, Intel i7-11800H (8 logical CPUs reported), 15.7 GB RAM.
- **NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096 MB**, driver 581.08. The earlier nvidia-smi summary truncated the name; the audit shows "3050 Ti".
- Python 3.11.9. Key versions: torch 2.7.1+cpu, torchvision 0.22.1+cpu, transformers 4.57.3, huggingface_hub 0.36.2, open_clip_torch 3.3.0, pycocotools 2.0.11, numpy 2.4.6, pydantic 2.14.1.
- Free disk after the venv install: repo drive 4.1 GB, store (D:) 5.9 GB.

**Checkpoint dry-run** (commit SHAs resolved at the time of the run; the lock is written on the first real download):

| Name | Repo @ commit | Size |
|---|---|---|
| grounding-dino-tiny | `IDEA-Research/grounding-dino-tiny@a2bb814dd3…` | 658.3 MB |
| remoteclip-vit-b-32 | `chendelong/RemoteCLIP@bf1d8a3ccf…` | 577.2 MB |
| openai-clip-vit-b-32 | `timm/vit_base_patch32_clip_224.openai@a6f597a30f…` | 577.2 MB |

Total is 1.77 GB, which fits in the 5.93 GB free on D: with a 1 GB margin. **Nothing was downloaded yet;** downloading is deferred to M2/M3 so laptop disk stays free for the dataset subset.

**H0 (prompt length):** 80 raw names = **251 / 256 tokens**, so there is no truncation for raw names. Details are in `reports/M0_gap_hypotheses_novelty.md`.

## Experiment machine audit (audited 2026-10-20 17:05 UTC; commit 36a5d2b by `Amit-adh`; reviewed 2026-10-21)
**Host:** `LAPTOP-5CPRTI27`, recorded in `outputs/env/LAPTOP-5CPRTI27.json` and `requirements-lock-LAPTOP-5CPRTI27.txt`. The audit ran at git `1bf7a57` with a clean tree.

| Item | Value |
|---|---|
| GPU | **NVIDIA GeForce RTX 5060 Laptop GPU**, compute capability **12.0** (Blackwell), 7.93 GB total (torch), driver 616.92 |
| CUDA stack | torch **2.7.1+cu128**, torchvision 0.22.1+cu128, CUDA build 12.8, `cuda_available: true`, cuDNN 90701 (9.7.1) |
| CPU / RAM | Intel Family 6 Model 198, 20 logical CPUs, 15.4 GB RAM |
| OS / Python | Windows build 10.0.26200; **Python 3.10.11** (the dev laptop has 3.11.9) |
| Disk | 191–192 GB free (repo and store) |
| Tests | `pytest` reported run by the user. The pass count is **not captured in the repo** (env_audit does not record it) |

### Cross-machine version check
These are identical on both machines: transformers 4.57.3, huggingface_hub 0.36.2, safetensors 0.8.0, open_clip_torch 3.3.0, pycocotools 2.0.11, pillow 12.3.0, pyyaml 6.0.3, pydantic 2.14.1, psutil 7.2.2, pytest 9.1.1.

**Drift found and fixed.** With Python 3.10, pip had resolved numpy 2.2.6 / pandas 2.3.3 / matplotlib 3.10.9 on the RTX 5060, while the laptop had 2.4.6 / 3.0.6 / 3.11.2. pandas 3.x changes defaults (copy-on-write, string dtype), so the same code could behave differently.
- Fix: the RTX 5060 is the reference environment for all reported results. `requirements.txt` now pins these three packages, and every package that already matched, to exact versions.
- The laptop was reinstalled to match, and **16/16 tests pass** on it after the change. Its lock file and env audit were refreshed.

**Remaining differences, all accepted:**
- the torch build (`+cpu` vs `+cu128`; intended);
- `exceptiongroup` and `tomli` (Python 3.10 backports);
- minor transitive versions of contourpy, fonttools, networkx and wcwidth (plotting/utility dependencies, not used in metric computation);
- the Python minor version (3.10 vs 3.11).

**Anomaly to watch:** the two GPU memory readings disagree.
- torch reported 6.88 GB free.
- nvidia-smi reported 7081 / 8151 MB used. This may be a WDDM accounting quirk, or another process may have been using the GPU.
- Consequence: before any latency or memory measurement (D9), confirm the GPU is idle and record `torch.cuda.mem_get_info()` at run start. Build this into the M12 profiling.

## Known limitations
- The RTX 5060's `pytest` result is not recorded in the repo (reported by the user only).
- The dev laptop has CPU-only torch. The 4 GB GPU is deliberately unused, because it is shared with games and is a different architecture from the experiment GPU.
- Determinism is best-effort (`warn_only=True`); some CUDA ops are nondeterministic.
- Commits use each machine's global git identity: `Himanshu-RG` on the dev laptop, `Amit-adh` on the RTX 5060 machine.

## Exit criteria
| Criterion | Status |
|---|---|
| Environment recorded | ✅ both machines (`LAPTOP-8N8B1VIP`, `LAPTOP-5CPRTI27`) |
| Reproducible dependencies | ✅ exact pins aligned to the RTX 5060; lock files for both machines |
| Project structure established, nothing destroyed | ✅ |
| Tests pass | ✅ 16/16 |
| Checkpoint pinning | ✅ commits resolved; sha256 lock written on first download |

## Next: Module 2 (dataset selection & validation)
Depends on M0 (protocol decisions) and M1. Steps:
1. Download the LAE-80C JSON (45 MB) and resolve the image count (3,592 vs. ~20k).
2. Build the manifest.
3. Validate boxes and categories.
4. Freeze the dev/test split.
5. Decide the class-naming policy, and re-run the H0 token check if names change.
6. Decide the VisDrone mapping.
