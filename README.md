# Adaptive Vocabulary Selection for Open-Vocabulary Detection in Aerial Imagery

Research code (Team Transmontane). We study whether choosing a small, image-relevant vocabulary before running a **frozen Grounding DINO** improves accuracy, recall, and efficiency on aerial imagery (primary benchmark: LAE-80C).

> **Start with [CONTEXT.md](CONTEXT.md).** It is the project's ground truth: goals, rules, frozen decisions, status, and results. Per-module reports live in [reports/](reports/).

## Status
M0–M11 and the one-time test evaluation are done.
- **Final test** (2,862 held-out images, everything fixed in advance): the frozen adaptive RemoteCLIP selector raises Grounding DINO-T AP50 from 0.0053 to 0.0280 (standard) and from 0.0157 to 0.0418 (source-aware), robust under both. It also robustly beats a fixed top-8 selector.
- The gain is on DIOR/DOTA-like data, plus NWPU VHR-10 (0.088 → 0.129). There is none on FAIR1M/xView.
- In progress: M12 idle-GPU latency, M13 reproducibility, M14 paper.

Further reading:
- Results: [reports/FINAL_TEST.md](reports/FINAL_TEST.md)
- Overview for reviewers: [reports/PROJECT_OVERVIEW.md](reports/PROJECT_OVERVIEW.md)
- Short summary: [reports/PROJECT_SUMMARY.md](reports/PROJECT_SUMMARY.md)
- **Reproducing everything: [REPRODUCE.md](REPRODUCE.md)**, including `python scripts/verify_artifacts.py`

## Two-machine setup
| Machine | Use | Torch wheel |
|---|---|---|
| Dev laptop (RTX 3050 Ti, 4 GB; CPU torch only) | code, unit tests, CPU smoke tests on a data subset | CPU |
| Experiment machine (RTX 5060, 8 GB) | all reported experiments and all latency numbers | **cu128** (Blackwell needs torch ≥ 2.7 + CUDA 12.8) |

Code, configs, small manifests, metrics, and env audits are synced through git. Datasets, checkpoints, and raw predictions live in a per-machine **store** outside the repo.

## Install
```bash
# 1. venv (put it in the store, not in OneDrive)
python -m venv <store>/venv
<store>/venv/Scripts/activate            # Windows: <store>\venv\Scripts\activate

# 2. torch -- pick ONE
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu    # dev laptop
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128  # RTX 5060

# 3. everything else + this package
pip install -r requirements.txt        # or requirements-gpu.txt on the GPU machine
pip install -e .
python scripts/env_audit.py --write-lock   # env JSON + UTF-8 lock file (don't use `pip freeze >` in PowerShell: it writes UTF-16)

# 4. tell the code where the store is
cp configs/paths.example.yaml configs/paths.local.yaml   # edit `store:`; or set AOVD_STORE
```

## Verify the environment
```bash
pytest                                   # unit tests
python scripts/env_audit.py              # writes outputs/env/<hostname>.json (commit it)
python scripts/fetch_checkpoints.py --dry-run   # resolves pinned commits + sizes, checks disk
python scripts/fetch_checkpoints.py      # downloads into <store>/checkpoints, writes configs/checkpoints.lock.json
```
Set `HF_HOME=<store>/hf_cache` to keep the Hugging Face cache off a small system drive.

## Layout
```
configs/        YAML configs, checkpoint list + lock file
data/manifests/ dataset manifests and frozen split ID lists
src/aerial_ovd/ data, detector, vocabulary, evaluation, experiments, utils
scripts/        command-line entry points
tests/          pytest suite
outputs/        env audits, metrics, figures (raw predictions are not committed)
reports/        per-module reports
```

## Rules this code follows
No test-set leakage (tuning on the dev split only); the detector stays frozen; every run records its config hash, seed, versions, and hardware; failed images are logged and stay in the denominator. See CONTEXT.md §3.
