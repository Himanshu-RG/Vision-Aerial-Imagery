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
| `requirements-lock-LAPTOP-8N8B1VIP.txt` | `pip freeze` of the dev-laptop venv (50 packages) |
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

## Known limitations
- The experiment machine (RTX 5060) has **not been audited yet**. Someone must run the install with the cu128 wheels plus `scripts/env_audit.py` there and commit `outputs/env/<host>.json` and `requirements-lock-<host>.txt`.
- The dev laptop has CPU-only torch. The 4 GB GPU is deliberately unused, because it is shared with games and is a different architecture from the experiment GPU.
- Determinism is best-effort (`warn_only=True`); some CUDA ops are nondeterministic.
- The GitHub remote is not configured. Nothing is pushed until the team provides the repo URL.
- The git identity used for commits is the machine's global config (`Himanshu-RG`).

## Exit criteria
| Criterion | Status |
|---|---|
| Environment recorded | ✅ dev laptop; ⏳ RTX 5060 pending |
| Reproducible dependencies | ✅ pinned + lock file (laptop); ⏳ GPU lock pending |
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
