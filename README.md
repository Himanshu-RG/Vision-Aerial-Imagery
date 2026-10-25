# Adaptive Vocabulary Selection for Open-Vocabulary Detection in Aerial Imagery

Research code (Team Transmontane). We study whether choosing a small, image-relevant vocabulary before running a **frozen Grounding DINO** improves accuracy, recall, and efficiency on aerial imagery (primary benchmark: LAE-80C).

> **Start with [CONTEXT.md](CONTEXT.md).** It is the project's ground truth: goals, rules, frozen decisions, status, and results. Per-module reports live in [reports/](reports/).

## Status
M0 (literature), M1 (environment, both machines) and M2 (dataset: LAE-80C validated, split frozen, evaluation protocol defined) are done. M3 (Grounding DINO pipeline, validated evaluator) is done, and the E0 headroom check ran on the RTX 5060 (2026-10-25): the oracle vocabulary raises AP50@900 from 0.015 to 0.140 on 200 dev images (oracle diagnostic, `reports/E0_headroom.md`). Next: M4 (deployable selectors + simple baselines).

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
