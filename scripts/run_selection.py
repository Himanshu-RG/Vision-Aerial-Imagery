"""Score every image of a split with each vocabulary selector (no detector involved).

Usage:
    python scripts/run_selection.py --split dev --device cpu
    python scripts/run_selection.py --split dev --device cuda --selectors clip_openai_b32 remoteclip_b32         --out-dir outputs/selection_gpu/dev      # GPU timing run; never overwrites the committed inputs
Writes outputs/selection/<split>/<selector>.npz with:
    image_ids (N,), cat_ids (C,), scores (N, C), image_ms (N,) [selector time per image, excl. disk I/O],
    plus meta (JSON string: device, templates, text-cache time, host, git commit).
CPU timings on the dev laptop are NOT latency results (D9); GPU selector timing is measured on the RTX 5060.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aerial_ovd.utils.paths import get_store  # noqa: E402
from aerial_ovd.vocabulary.names import prompt_name  # noqa: E402
from aerial_ovd.vocabulary.selectors import (  # noqa: E402
    DEFAULT_TEMPLATES, CLIPSelector, FrequencySelector, RandomSelector,
)
from run_detection import load_manifest  # noqa: E402

MAN = REPO_ROOT / "data" / "manifests"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--selectors", nargs="*", default=["frequency", "random_s0", "random_s1", "random_s2",
                                                         "clip_openai_b32", "remoteclip_b32"])
    ap.add_argument("--limit", type=int, default=None, help="first N images only (smoke tests)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default outputs/selection/<split> (the committed selection inputs). Use a different "
                         "folder for timing runs, e.g. outputs/selection_gpu/<split>, so they never overwrite them")
    args = ap.parse_args()
    if args.split == "test":
        print("Selection on TEST is only allowed for the final, frozen evaluation (rule 3).", file=sys.stderr)

    store = get_store()
    rows = [r for r in load_manifest() if r["split"] == args.split]
    rows.sort(key=lambda r: int(r["image_id"]))
    if args.limit:
        rows = rows[: args.limit]
    vocab = list(csv.DictReader((MAN / "lae80c_vocabulary.csv").open(encoding="utf-8")))
    cat_ids = np.array([int(v["category_id"]) for v in vocab])
    names = [prompt_name(v["raw_name"], "light") for v in vocab]
    dev_freq = np.array([int(v["n_images_dev"]) for v in vocab], dtype=float)  # dev prior only

    ck = store / "checkpoints"
    builders = {
        "frequency": lambda: FrequencySelector(dev_freq),
        "random_s0": lambda: RandomSelector(len(cat_ids), 0),
        "random_s1": lambda: RandomSelector(len(cat_ids), 1),
        "random_s2": lambda: RandomSelector(len(cat_ids), 2),
        "clip_openai_b32": lambda: CLIPSelector("clip_openai_b32", "ViT-B-32",
                                                ck / "openai-clip-vit-b-32" / "open_clip_pytorch_model.bin",
                                                names, device=args.device),
        "remoteclip_b32": lambda: CLIPSelector("remoteclip_b32", "ViT-B-32",
                                               ck / "remoteclip-vit-b-32" / "RemoteCLIP-ViT-B-32.pt",
                                               names, device=args.device, state_dict_format=True),
    }
    from PIL import Image

    out_dir = args.out_dir or (REPO_ROOT / "outputs" / "selection" / args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    git_dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs"],
                                    capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip())
    img_dir = store / "datasets" / "LAE-80C-images"
    for sel_name in args.selectors:
        t0 = time.perf_counter()
        sel = builders[sel_name]()
        load_s = time.perf_counter() - t0
        scores = np.zeros((len(rows), len(cat_ids)))
        ms = np.zeros(len(rows))
        failed = []
        for i, r in enumerate(rows):
            try:
                with Image.open(img_dir / r["file_name"]) as im:
                    image = im.convert("RGB")  # disk I/O excluded from selector time
                t = time.perf_counter()
                scores[i] = sel.scores(image)
                ms[i] = (time.perf_counter() - t) * 1e3
            except Exception as e:  # noqa: BLE001 - recorded, never silently skipped
                failed.append({"image_id": int(r["image_id"]), "error": repr(e)})
                scores[i] = np.nan
        meta = {"selector": sel_name, "split": args.split, "device": args.device, "host": socket.gethostname(),
                "git_commit": git, "git_dirty_code": git_dirty, "n_images": len(rows), "failed": failed, "model_load_s": round(load_s, 2),
                "templates": list(DEFAULT_TEMPLATES) if sel_name.startswith(("clip", "remoteclip")) else None,
                "text_cache_ms": getattr(sel, "text_cache_ms", None),
                "timing_note": "selector time per image excludes disk I/O; CPU timings are not latency results (D9)"}
        np.savez_compressed(out_dir / f"{sel_name}.npz", image_ids=np.array([int(r["image_id"]) for r in rows]),
                            cat_ids=cat_ids, scores=scores, image_ms=ms, meta=json.dumps(meta))
        print(f"{sel_name:<18} images={len(rows)} failed={len(failed)} median_ms={np.median(ms):.1f} load_s={load_s:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
