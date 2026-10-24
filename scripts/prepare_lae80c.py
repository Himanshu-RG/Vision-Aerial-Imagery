"""Module 2: clean LAE-80C annotations, build the manifest, freeze the dev/test split.

Usage:
    python scripts/prepare_lae80c.py            # needs the annotation JSON in <store>/datasets/LAE-1M
    python scripts/prepare_lae80c.py --download # fetch the annotation JSON first (45 MB, pinned commit)

Outputs (committed, small):
    data/manifests/lae80c_manifest.csv          one row per image, incl. split
    data/manifests/lae80c_dev_ids.txt           frozen dev image ids   (tuning only)
    data/manifests/lae80c_test_ids.txt          frozen test image ids  (final evaluation only)
    data/manifests/lae80c_vocabulary.csv        id, raw name, prompt name, source, counts
    data/manifests/lae80c_prepare_report.json   cleaning log, stats, sha256 of inputs/outputs
Outputs (store, not committed):
    <store>/datasets/LAE-80C-clean/{all,dev,test}.json   cleaned COCO ground truth

Re-running must reproduce byte-identical committed files; the script refuses to overwrite a
frozen split that differs (pass --allow-resplit only with a new decision row in CONTEXT.md).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aerial_ovd.data.lae80c import (  # noqa: E402
    build_manifest, clean_coco, load_coco, source_label_space, subset_coco,
)
from aerial_ovd.data.split import iterative_stratification  # noqa: E402
from aerial_ovd.utils.config import load_yaml  # noqa: E402
from aerial_ovd.utils.paths import get_store  # noqa: E402
from aerial_ovd.vocabulary.names import build_prompt, prompt_name  # noqa: E402

CFG = REPO_ROOT / "configs" / "data_lae80c.yaml"
MAN = REPO_ROOT / "data" / "manifests"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--allow-resplit", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(CFG)
    ds, cl, sp = cfg["dataset"], cfg["cleaning"], cfg["split"]
    store = get_store()
    os.environ.setdefault("HF_HOME", str(store / "hf_cache"))
    raw_dir = store / ds["local_dir"]
    raw_json = raw_dir / ds["annotation_file"]

    if args.download or not raw_json.is_file():
        from huggingface_hub import hf_hub_download

        hf_hub_download(ds["hf_repo"], ds["annotation_file"], repo_type="dataset",
                        revision=ds["revision"], local_dir=raw_dir)
    raw = load_coco(raw_json)

    clean, rep = clean_coco(raw, clip_to_image=cl["clip_to_image"],
                            drop_exact_duplicates=cl["drop_exact_duplicates"],
                            drop_nonpositive=cl["drop_nonpositive"],
                            tiny_side_flag_px=cl["tiny_side_flag_px"])
    manifest = build_manifest(clean, tuple(cfg["size_bins_px2"]))

    # ---- frozen split -------------------------------------------------------------
    labels = {r["image_id"]: [int(c) for c in r["classes"].split()] for r in manifest}
    dev, test = iterative_stratification(labels, [sp["dev_fraction"], 1 - sp["dev_fraction"]], seed=sp["seed"])
    dev_set = set(dev)
    dev_txt = "".join(f"{i}\n" for i in sorted(dev))
    test_txt = "".join(f"{i}\n" for i in sorted(test))
    for name, txt in (("lae80c_dev_ids.txt", dev_txt), ("lae80c_test_ids.txt", test_txt)):
        p = MAN / name
        if p.is_file() and p.read_text(encoding="utf-8") != txt and not args.allow_resplit:
            print(f"REFUSING: {p} exists and differs from the recomputed split. The split is frozen.",
                  file=sys.stderr)
            return 2
    MAN.mkdir(parents=True, exist_ok=True)
    (MAN / "lae80c_dev_ids.txt").write_text(dev_txt, encoding="utf-8", newline="\n")
    (MAN / "lae80c_test_ids.txt").write_text(test_txt, encoding="utf-8", newline="\n")
    for r in manifest:
        r["split"] = "dev" if r["image_id"] in dev_set else "test"

    # ---- manifest + vocabulary ------------------------------------------------------
    cols = ["image_id", "file_name", "source", "split", "width", "height", "n_boxes", "n_classes",
            "classes", "n_small", "n_medium", "n_large", "n_tiny_flag"]
    with (MAN / "lae80c_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(manifest)

    space = source_label_space(clean)
    src_of_cat = {c: s for s, cs in space.items() for c in cs}
    box_count, img_count = Counter(), {"dev": Counter(), "test": Counter()}
    for a in clean["annotations"]:
        box_count[a["category_id"]] += 1
    for r in manifest:
        for c in r["classes"].split():
            img_count[r["split"]][int(c)] += 1
    vocab_rows = []
    for c in sorted(clean["categories"], key=lambda c: c["id"]):
        vocab_rows.append({"category_id": c["id"], "raw_name": c["name"], "prompt_name": prompt_name(c["name"]),
                           "source": src_of_cat.get(c["id"], ""), "n_boxes": box_count[c["id"]],
                           "n_images_dev": img_count["dev"][c["id"]], "n_images_test": img_count["test"][c["id"]]})
    with (MAN / "lae80c_vocabulary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(vocab_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(vocab_rows)

    # ---- cleaned GT to the store ------------------------------------------------------
    out_dir = store / "datasets" / "LAE-80C-clean"
    write_json(out_dir / "all.json", clean)
    write_json(out_dir / "dev.json", subset_coco(clean, dev_set))
    write_json(out_dir / "test.json", subset_coco(clean, set(test)))

    # ---- report -------------------------------------------------------------------------
    by_src = defaultdict(lambda: Counter())
    for r in manifest:
        by_src[r["source"]][r["split"]] += 1
    n = len(manifest)
    stats = {
        "n_images": n,
        "n_boxes": sum(r["n_boxes"] for r in manifest),
        "images_per_source_and_split": {s: dict(c) for s, c in sorted(by_src.items())},
        "boxes_per_image": {"mean": round(sum(r["n_boxes"] for r in manifest) / n, 2),
                            "max": max(r["n_boxes"] for r in manifest)},
        "classes_per_image_mean": round(sum(r["n_classes"] for r in manifest) / n, 3),
        "size_bins_share": {k: round(sum(r[f"n_{k}"] for r in manifest) / sum(r["n_boxes"] for r in manifest), 4)
                            for k in ("small", "medium", "large")},
        "classes_missing_from_dev": [v["raw_name"] for v in vocab_rows if v["n_images_dev"] == 0],
        "classes_with_lt3_dev_images": {v["raw_name"]: v["n_images_dev"] for v in vocab_rows if v["n_images_dev"] < 3},
        "source_label_space": {s: sorted(cs) for s, cs in sorted(space.items())},
        "full_prompt_light_names": build_prompt([v["prompt_name"] for v in vocab_rows]),
    }
    report = {
        "config": cfg,
        "inputs": {"annotation_json_sha256": sha256(raw_json)},
        "cleaning": rep.to_dict(),
        "stats": stats,
        "outputs_sha256": {p.name: sha256(p) for p in sorted(MAN.glob("lae80c_*")) if p.suffix in (".csv", ".txt")},
        "store_outputs_sha256": {p.name: sha256(p) for p in sorted(out_dir.glob("*.json"))},
    }
    write_json(MAN / "lae80c_prepare_report.json", report)
    print(json.dumps({"cleaning": rep.to_dict()["counts"], "stats": {k: v for k, v in stats.items()
                      if k not in ("source_label_space", "full_prompt_light_names")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
