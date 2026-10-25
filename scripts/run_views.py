"""Module 5: score the whole image and all crops with a CLIP model; store per-view logits.

Usage:
    python scripts/run_views.py --split dev --model remoteclip_b32 [--device cuda] [--limit N] [--out-dir DIR]
Writes <out-dir or outputs/selection/<split>>/views_<model>.npz:
    image_ids (N,), cat_ids (C,), global_logits (N, C), crop_logits (N, V, C), crop_grid_ids (V,),
    image_ms (N,) [all views of one image, excl. disk I/O], meta (JSON).
All images in a split are assumed to yield the same number of crops (true for the fixed grids).
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
    DEFAULT_GRIDS, DEFAULT_OVERLAP, DEFAULT_TEMPLATES, CLIPSelector, view_logits,
)
from run_detection import load_manifest  # noqa: E402

MODELS = {
    "clip_openai_b32": ("openai-clip-vit-b-32/open_clip_pytorch_model.bin", False),
    "remoteclip_b32": ("remoteclip-vit-b-32/RemoteCLIP-ViT-B-32.pt", True),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--grids", type=int, nargs="+", default=list(DEFAULT_GRIDS),
                    help="crop grids to compute (timing runs: only the grids the chosen variant uses, e.g. 2 3)")
    args = ap.parse_args()

    store = get_store()
    rows = sorted((r for r in load_manifest() if r["split"] == args.split), key=lambda r: int(r["image_id"]))
    if args.limit:
        rows = rows[: args.limit]
    vocab = list(csv.DictReader((REPO_ROOT / "data" / "manifests" / "lae80c_vocabulary.csv").open(encoding="utf-8")))
    cat_ids = np.array([int(v["category_id"]) for v in vocab])
    names = [prompt_name(v["raw_name"], "light") for v in vocab]
    weights, sd = MODELS[args.model]
    t0 = time.perf_counter()
    sel = CLIPSelector(args.model, "ViT-B-32", store / "checkpoints" / weights, names, device=args.device,
                       state_dict_format=sd)
    load_s = time.perf_counter() - t0

    from PIL import Image

    import torch

    G, Cr, ms, failed, gids = [], [], [], [], None
    for i, r in enumerate(rows):
        with Image.open(store / "datasets" / "LAE-80C-images" / r["file_name"]) as im:
            image = im.convert("RGB")
        try:
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t = time.perf_counter()
            g, c, gi, _ = view_logits(sel, image, grids=tuple(args.grids))
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            ms.append((time.perf_counter() - t) * 1e3)
            if gids is None:
                gids = gi
            elif not np.array_equal(gids, gi):
                raise RuntimeError("crop layout differs between images")
            G.append(g)
            Cr.append(c)
        except Exception as e:  # noqa: BLE001 - recorded; NaN rows keep the image in the denominator
            failed.append({"image_id": int(r["image_id"]), "error": repr(e)})
            G.append(np.full(len(cat_ids), np.nan, np.float32))
            Cr.append(None)
            ms.append(np.nan)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}  median ms/img {np.nanmedian(ms):.0f}", flush=True)
    V = len(gids)
    Cr = np.stack([c if c is not None else np.full((V, len(cat_ids)), np.nan, np.float32) for c in Cr])
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs"],
                                capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip())
    meta = {"model": args.model, "split": args.split, "device": args.device, "host": socket.gethostname(),
            "git_commit": git, "git_dirty_code": dirty, "grids": list(args.grids), "overlap": DEFAULT_OVERLAP,
            "n_crops": int(V), "templates": list(DEFAULT_TEMPLATES), "logit_scale": float(sel.model.logit_scale.exp()),
            "model_load_s": round(load_s, 2), "failed": failed,
            "timing_note": "image_ms = all 1 + n_crops views of one image (batched), excl. disk I/O; CPU is not a latency result"}
    out_dir = args.out_dir or (REPO_ROOT / "outputs" / "selection" / args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"views_{args.model}.npz", image_ids=np.array([int(r["image_id"]) for r in rows]),
                        cat_ids=cat_ids, global_logits=np.stack(G), crop_logits=Cr, crop_grid_ids=gids,
                        image_ms=np.array(ms), meta=json.dumps(meta))
    print(f"{args.model}: images={len(rows)} failed={len(failed)} crops/img={V} median_ms={np.nanmedian(ms):.0f} "
          f"logit_scale={meta['logit_scale']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
