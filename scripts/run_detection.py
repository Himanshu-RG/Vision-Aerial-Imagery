"""Run frozen Grounding DINO on a frozen image sample under several vocabulary conditions.

Usage:
    python scripts/run_detection.py --config configs/experiments/e0_headroom.yaml
    python scripts/run_detection.py --config ... --device cpu --n-images 5     # smoke test
    python scripts/run_detection.py --config ... --resume <run_dir>            # continue a run

Writes outputs/predictions/<run_id>/:
    run_config.json     resolved config, env snapshot, image sample, checkpoint lock entry
    <condition>.jsonl   one record per image (detections, vocabulary, timings, status)
    log.jsonl           structured log (failures, OOMs)
Records are appended as they finish, so an interrupted run resumes without redoing images.
Failed images are recorded (status "failed") and stay in the evaluation denominator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aerial_ovd.data.split import iterative_stratification  # noqa: E402
from aerial_ovd.utils.config import load_yaml  # noqa: E402
from aerial_ovd.utils.logging import get_logger  # noqa: E402
from aerial_ovd.utils.paths import get_store  # noqa: E402
from aerial_ovd.utils.run_id import make_run_id  # noqa: E402
from aerial_ovd.utils.seed import set_seed  # noqa: E402
from aerial_ovd.vocabulary.conditions import vocabulary_for  # noqa: E402
from aerial_ovd.vocabulary.names import prompt_name  # noqa: E402

MAN = REPO_ROOT / "data" / "manifests"


def load_manifest() -> list[dict]:
    with (MAN / "lae80c_manifest.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_vocab(policy: str) -> tuple[dict[int, str], dict[int, str]]:
    """Prompt names under the naming policy (D14) and the source of each category id."""
    prompt, source = {}, {}
    with (MAN / "lae80c_vocabulary.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            prompt[int(r["category_id"])] = prompt_name(r["raw_name"], policy)
            source[int(r["category_id"])] = r["source"]
    return prompt, source


def sample_images(rows: list[dict], split: str, n: int | None, seed: int) -> list[int]:
    """Deterministic class-stratified sample of `n` images from a split (all if n is None)."""
    pool = {int(r["image_id"]): [int(c) for c in r["classes"].split()] for r in rows if r["split"] == split}
    if n is None or n >= len(pool):
        return sorted(pool)
    frac = n / len(pool)
    picked, rest = iterative_stratification(pool, [frac, 1 - frac], seed=seed)
    # Stratification hits the target only approximately; trim or top up deterministically to exactly n.
    rng = random.Random(seed)
    if len(picked) > n:
        picked = rng.sample(sorted(picked), n)
    elif len(picked) < n:
        picked = list(picked) + rng.sample(sorted(rest), n - len(picked))
    return sorted(picked)


def done_ids(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    ok = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["status"] == "ok":
                ok.add(rec["image_id"])
    return ok


def gpu_snapshot(torch) -> dict:
    snap = {}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        snap["torch_free_gb"], snap["torch_total_gb"] = round(free / 1024**3, 2), round(total / 1024**3, 2)
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            used, util, power = [x.strip() for x in out.stdout.strip().split(",")]
            snap.update({"smi_used_mb": float(used), "smi_util_pct": float(util), "smi_power_w": power})
        except Exception as e:  # noqa: BLE001
            snap["smi_error"] = str(e)
    return snap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--device", default=None, help="override detector.device")
    ap.add_argument("--n-images", type=int, default=None, help="override n_images (smoke tests)")
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.device:
        cfg["detector"]["device"] = args.device
    if args.n_images is not None:
        cfg["n_images"] = args.n_images
    det_cfg = cfg["detector"]

    import torch
    from PIL import Image

    from aerial_ovd.detector.gdino import GroundingDINO

    if det_cfg["device"].startswith("cuda") and not torch.cuda.is_available():
        print("config asks for CUDA but torch.cuda.is_available() is False -- wrong venv?", file=sys.stderr)
        return 2

    run_id = make_run_id(cfg["name"], cfg)
    run_dir = args.resume or (REPO_ROOT / "outputs" / "predictions" / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"run.{run_id}", run_dir / "log.jsonl")
    seed_rec = set_seed(cfg.get("seed", 0), deterministic=cfg.get("deterministic", False))

    store = get_store()
    rows = load_manifest()
    by_id = {int(r["image_id"]): r for r in rows}
    prompt_names, cat_source = load_vocab(cfg.get("naming_policy", "light"))
    all_ids = sorted(prompt_names)
    source_space = {}
    for iid, r in by_id.items():
        source_space[iid] = {c for c, s in cat_source.items() if s == r["source"]}
    gt_classes = {iid: [int(c) for c in r["classes"].split()] for iid, r in by_id.items()}
    image_ids = sample_images(rows, cfg["split"], cfg.get("n_images"), cfg["sample_seed"])

    # Precomputed deployable selector scores (outputs/selection/<split>/<name>.npz, committed).
    selector_scores, selector_meta = {}, {}
    for cond in cfg["conditions"]:
        if cond.startswith("sel:"):
            import numpy as np

            name = cond.split(":")[1]
            if name in selector_scores:
                continue
            npz = REPO_ROOT / "outputs" / "selection" / cfg["split"] / f"{name}.npz"
            z = np.load(npz, allow_pickle=False)
            if list(z["cat_ids"]) != sorted(int(c) for c in z["cat_ids"]):
                raise ValueError(f"{npz}: category ids are not sorted")
            selector_scores[name] = {int(i): z["scores"][j] for j, i in enumerate(z["image_ids"])}
            selector_meta[name] = {"file": npz.relative_to(REPO_ROOT).as_posix(), "meta": json.loads(str(z["meta"]))}

    lock = json.loads((REPO_ROOT / "configs" / "checkpoints.lock.json").read_text(encoding="utf-8"))
    ckpt_name = det_cfg["checkpoint"]
    meta = {
        "run_id": run_id, "config": cfg, "seed": seed_rec, "host": socket.gethostname(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                     cwd=REPO_ROOT).stdout.strip(),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                         cwd=REPO_ROOT).stdout.strip()),
        "torch": torch.__version__, "cuda": torch.version.cuda, "checkpoint_lock": lock.get(ckpt_name),
        "image_ids": image_ids, "gpu_at_start": gpu_snapshot(torch), "selectors": selector_meta,
        "latency_definition": "per (image, condition): preprocess (tokenise + positive map + image "
                              "resize/normalise) + forward + postprocess; excludes disk I/O and "
                              "model loading (reported separately as load_ms / model_load_s)",
    }

    t_load = time.perf_counter()
    det = GroundingDINO(store / "checkpoints" / ckpt_name, device=det_cfg["device"],
                        precision=det_cfg["precision"], num_select=det_cfg["num_select"],
                        class_reduce=det_cfg.get("class_reduce", "mean"))
    meta["model_load_s"] = round(time.perf_counter() - t_load, 2)
    (run_dir / "run_config.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    log.info("run started", extra={"run_id": run_id, "n_images": len(image_ids), "device": det_cfg["device"]})

    img_dir = store / "datasets" / "LAE-80C-images"
    conditions = cfg["conditions"]
    files = {c: run_dir / (c.replace(":", "__") + ".jsonl") for c in conditions}  # ":" is not allowed in Windows file names
    finished = {c: done_ids(p) for c, p in files.items()}

    # Warm-up (not recorded): first-call CUDA/cuDNN set-up would otherwise inflate latency.
    if det_cfg.get("warmup", 0) and image_ids:
        with Image.open(img_dir / by_id[image_ids[0]]["file_name"]) as im:
            warm = im.convert("RGB")
        for _ in range(det_cfg["warmup"]):
            det.detect(warm, [prompt_names[c] for c in all_ids])

    n_total = len(image_ids)
    for i, iid in enumerate(image_ids, 1):
        todo = [c for c in conditions if iid not in finished[c]]
        if not todo:
            continue
        r = by_id[iid]
        t0 = time.perf_counter()
        try:
            with Image.open(img_dir / r["file_name"]) as im:
                image = im.convert("RGB")
            load_ms = (time.perf_counter() - t0) * 1e3
        except Exception as e:  # noqa: BLE001
            for c in todo:
                rec = {"image_id": iid, "condition": c, "status": "failed", "error": f"load: {e!r}"}
                with files[c].open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            log.error("image load failed", extra={"image_id": iid, "error": repr(e)})
            continue
        for c in todo:
            vocab = vocabulary_for(c, iid, all_ids, gt_classes, source_space, seed=cfg.get("random_seed", 0),
                                   selector_scores=selector_scores)
            rec = {"image_id": iid, "condition": c, "vocab": vocab, "load_ms": round(load_ms, 2)}
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                d = det.detect(image, [prompt_names[k] for k in vocab])
                xyxy = d.boxes_xyxy
                rec.update({
                    "status": "ok", "n_text_tokens": d.n_text_tokens,
                    "timings_ms": {k: round(v, 2) for k, v in d.timings_ms.items()},
                    "latency_ms": round(sum(d.timings_ms.values()), 2),
                    "detections": [[vocab[int(ci)], round(float(s), 5), round(float(b[0]), 2), round(float(b[1]), 2),
                                    round(float(b[2] - b[0]), 2), round(float(b[3] - b[1]), 2)]
                                   for ci, s, b in zip(d.class_index, d.scores, xyxy)],
                })
                if torch.cuda.is_available():
                    rec["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                rec.update({"status": "failed", "error": f"cuda_oom: {e}"})
                log.error("CUDA OOM", extra={"image_id": iid, "condition": c})
            except Exception as e:  # noqa: BLE001
                rec.update({"status": "failed", "error": repr(e)})
                log.error("detection failed", extra={"image_id": iid, "condition": c, "error": repr(e)})
            with files[c].open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        if i % 10 == 0 or i == n_total:
            log.info(f"progress {i}/{n_total}", extra={"image_id": iid})

    meta["gpu_at_end"] = gpu_snapshot(torch)
    (run_dir / "run_config.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    log.info("run finished", extra={"run_dir": str(run_dir)})
    print(run_dir)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
