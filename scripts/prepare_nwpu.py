"""M11 part 2 (D45): fetch NWPU VHR-10 (HF mirror, pinned revision) and convert it to the project's formats.

Usage:
    python scripts/prepare_nwpu.py
Writes (store, not committed; research-use-only images): <store>/datasets/NWPU-VHR10/{images/, nwpu.json}
Writes (committed): data/manifests/nwpu_manifest.csv, data/manifests/nwpu_prepare_report.json
* The 650 positive images only. Boxes "(x1,y1),(x2,y2),a" -> COCO xywh, NWPU class -> LAE-80C id (NWPU_TO_LAE).
* Ship (class 2) has no single equivalent in the 80-class vocabulary: its boxes are dropped (count reported).
* Colour-infrared (CIR, Vaihingen-derived) flag: images 421-453 (33 positive images). Found with a colour rule
  (share of red pixels R > G+20 & R > B+20 >= 0.05 AND of cyan pixels G > R+15 & B > R+15 >= 0.05) and confirmed by
  visual inspection: the rule also flagged 640 (natural colour; excluded) and missed 442 (CIR; included), leaving
  exactly the contiguous block 421-453. Vaihingen is in RemoteCLIP's training data (D44): reported as a subgroup.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aerial_ovd.data.splits import NWPU_ID_OFFSET, NWPU_NAMES, NWPU_SOURCE, NWPU_TO_LAE, gt_path, image_dir  # noqa: E402
from aerial_ovd.utils.paths import get_store  # noqa: E402

REPO_ID = "SGhazalS/nwpu-vhr10"
REVISION = "f7aa66718fc693b13dc2a3218568b7817a18794b"
CIR_IDS = set(range(421, 454))
BOX = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*(\d+)")


def cir_share(im) -> float:
    a = np.asarray(im.convert("RGB").resize((256, 256)), dtype=np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return float(((r > g + 20) & (r > b + 20)).mean())


def main() -> int:
    from huggingface_hub import snapshot_download
    from PIL import Image

    store = get_store()
    raw = store / "datasets" / "NWPU-VHR10" / "raw"
    import time

    have = raw / "NWPU VHR-10 dataset"
    complete = ((have / "ground truth").is_dir() and len(list((have / "ground truth").glob("*.txt"))) == 650
                and len(list((have / "positive image set").glob("*"))) == 650)
    # Skip the network when the pinned files are present. Anonymous downloads of ~1,300 small files hit HTTP 429,
    # so the download resumes with backoff.
    for attempt in ([] if complete else range(12)):
        try:
            snapshot_download(REPO_ID, repo_type="dataset", revision=REVISION, local_dir=raw, max_workers=2,
                              allow_patterns=["NWPU VHR-10 dataset/positive image set/*", "NWPU VHR-10 dataset/ground truth/*",
                                              "NWPU VHR-10 dataset/readme.txt"])
            break
        except Exception as e:  # noqa: BLE001
            transient = "429" in str(e) or type(e).__name__ in ("LocalEntryNotFoundError", "HfHubHTTPError")
            if not transient or attempt == 11:
                raise
            print(f"rate limited, retrying in {60 * (attempt + 1)} s", flush=True)
            time.sleep(60 * (attempt + 1))
    base = raw / "NWPU VHR-10 dataset"
    out_img = image_dir(store, "nwpu")
    out_img.mkdir(parents=True, exist_ok=True)
    images, anns, rows, dropped, per_class, cir = [], [], [], Counter(), Counter(), []
    gt_files = sorted((base / "ground truth").glob("*.txt"))
    assert len(gt_files) == 650, len(gt_files)
    for gf in gt_files:
        stem = gf.stem
        src_img = next(p for p in (base / "positive image set").iterdir() if p.stem == stem)
        iid = NWPU_ID_OFFSET + int(stem)
        fname = f"{stem}{src_img.suffix.lower()}"
        shutil.copyfile(src_img, out_img / fname)
        with Image.open(src_img) as im:
            W, H = im.size
            share = cir_share(im)
        is_cir = int(stem) in CIR_IDS
        cir.append((stem, share))
        images.append({"id": iid, "file_name": fname, "width": W, "height": H})
        cls, sizes = [], Counter()
        for line in gf.read_text(encoding="utf-8", errors="replace").splitlines():
            m = BOX.search(line)
            if not m:
                continue
            x1, y1, x2, y2, a = map(int, m.groups())
            if a not in NWPU_TO_LAE:
                dropped[NWPU_NAMES.get(a, str(a))] += 1
                continue
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                dropped["degenerate"] += 1
                continue
            c = NWPU_TO_LAE[a]
            anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": c, "bbox": [x1, y1, w, h],
                         "area": float(w * h), "iscrowd": 0})
            per_class[NWPU_NAMES[a]] += 1
            cls.append(c)
            sizes["small" if w * h < 1024 else "medium" if w * h < 9216 else "large"] += 1
        if not cls:  # ship-only images have no evaluable object after the ship drop
            images.pop()
            (out_img / fname).unlink()
            dropped["images_ship_only"] += 1
            continue
        rows.append({"image_id": iid, "file_name": fname, "source": NWPU_SOURCE, "split": "nwpu", "width": W, "height": H,
                     "n_boxes": len(cls), "n_classes": len(set(cls)), "classes": " ".join(str(c) for c in sorted(set(cls))),
                     "n_small": sizes["small"], "n_medium": sizes["medium"], "n_large": sizes["large"], "n_tiny_flag": 0,
                     "cir": int(is_cir)})
    cats = json.loads((store / "datasets" / "LAE-80C-clean" / "dev.json").read_text(encoding="utf-8"))["categories"]
    gp = gt_path(store, "nwpu")
    gp.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}), encoding="utf-8")
    man = REPO_ROOT / "data" / "manifests" / "nwpu_manifest.csv"
    with man.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    rep = {"repo": REPO_ID, "revision": REVISION, "n_positive_images": len(gt_files), "n_images_kept": len(rows),
           "n_boxes_kept": len(anns), "boxes_per_class": dict(per_class), "dropped": dict(dropped),
           "cir_rule": "images 421-453 (colour rule + visual check; see docstring)", "n_cir": sum(r["cir"] for r in rows),
           "cir_share_quantiles": [round(float(q), 3) for q in np.quantile([s for _, s in cir], [0.5, 0.8, 0.85, 0.9, 0.95])],
           "gt_json_sha256": hashlib.sha256(gp.read_bytes()).hexdigest(),
           "manifest_sha256": hashlib.sha256(man.read_bytes()).hexdigest()}
    (REPO_ROOT / "data" / "manifests" / "nwpu_prepare_report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
