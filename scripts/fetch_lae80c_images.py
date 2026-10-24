"""Download LAE-80C images (pinned commit), extract all or one split, and validate every image.

Usage:
    python scripts/fetch_lae80c_images.py --split all  --delete-zip   # RTX 5060 machine
    python scripts/fetch_lae80c_images.py --split dev  --delete-zip   # dev laptop (small disk)
    python scripts/fetch_lae80c_images.py --split dev --validate-only  # re-check extracted images

Images go to <store>/datasets/LAE-80C-images/<file_name>.
Validation (written to data/manifests/lae80c_images_check_<host>_<split>.json, committed):
missing from zip, unreadable/corrupt, and width/height different from the annotation JSON.
Nothing is skipped silently: every problem is listed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import zipfile
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aerial_ovd.utils.config import load_yaml  # noqa: E402
from aerial_ovd.utils.paths import get_store  # noqa: E402

CFG = REPO_ROOT / "configs" / "data_lae80c.yaml"
MAN = REPO_ROOT / "data" / "manifests"
ZIP_BYTES = 1_970_281_656  # size of images.zip at the pinned revision (HF metadata)


def wanted_files(split: str) -> dict[str, tuple[int, int]]:
    """file_name -> (width, height) from the committed manifest."""
    import csv

    out = {}
    with (MAN / "lae80c_manifest.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if split == "all" or r["split"] == split:
                out[r["file_name"]] = (int(r["width"]), int(r["height"]))
    return out


def validate(img_dir: Path, wanted: dict[str, tuple[int, int]]) -> dict:
    from PIL import Image

    problems = {"missing": [], "unreadable": [], "size_mismatch": []}
    for name, (w, h) in sorted(wanted.items()):
        p = img_dir / name
        if not p.is_file():
            problems["missing"].append(name)
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:  # verify() invalidates the handle; reopen to read size + decode
                im.load()
                if im.size != (w, h):
                    problems["size_mismatch"].append({"file": name, "json": [w, h], "image": list(im.size)})
        except Exception as e:  # noqa: BLE001 - every failure is recorded, none is fatal
            problems["unreadable"].append({"file": name, "error": f"{type(e).__name__}: {e}"})
    return {"n_expected": len(wanted), "n_ok": len(wanted) - sum(len(v) for v in problems.values()),
            "problems": problems}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test", "all"], required=True)
    ap.add_argument("--delete-zip", action="store_true", help="remove images.zip after extraction")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    ds = load_yaml(CFG)["dataset"]
    store = get_store()
    os.environ.setdefault("HF_HOME", str(store / "hf_cache"))
    img_dir = store / "datasets" / "LAE-80C-images"
    zip_path = store / ds["local_dir"] / ds["images_zip"]
    wanted = wanted_files(args.split)
    record = {"host": socket.gethostname(), "split": args.split, "revision": ds["revision"]}

    if not args.validate_only:
        todo = [n for n in wanted if not (img_dir / n).is_file()]
        if todo:
            if not zip_path.is_file():
                free = shutil.disk_usage(store).free
                if free < ZIP_BYTES + 1.5 * 1024**3:
                    print(f"not enough disk: {free/1024**3:.1f} GB free, need ~{ZIP_BYTES/1024**3 + 1.5:.1f} GB",
                          file=sys.stderr)
                    return 1
                from huggingface_hub import hf_hub_download

                print("downloading images.zip (~1.97 GB) ...")
                hf_hub_download(ds["hf_repo"], ds["images_zip"], repo_type="dataset",
                                revision=ds["revision"], local_dir=store / ds["local_dir"])
            h = hashlib.sha256()
            with zip_path.open("rb") as f:
                while b := f.read(1 << 22):
                    h.update(b)
            record["zip_sha256"] = h.hexdigest()
            with zipfile.ZipFile(zip_path) as zf:
                by_base = {}
                for info in zf.infolist():
                    if not info.is_dir():
                        by_base.setdefault(PurePosixPath(info.filename).name, info)
                record["zip_members"] = len(by_base)
                record["zip_members_not_in_annotations"] = sorted(set(by_base) - set(wanted_files("all")))[:20]
                img_dir.mkdir(parents=True, exist_ok=True)
                for i, name in enumerate(todo, 1):
                    info = by_base.get(name)
                    if info is None:
                        continue  # reported as missing by validate()
                    with zf.open(info) as src, (img_dir / name).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    if i % 500 == 0:
                        print(f"  extracted {i}/{len(todo)}")
            if args.delete_zip:
                zip_path.unlink()
                record["zip_deleted"] = True

    record["validation"] = validate(img_dir, wanted)
    out = MAN / f"lae80c_images_check_{record['host']}_{args.split}.json"
    if out.is_file():
        # A re-run with images already extracted does not touch the zip; keep the zip facts
        # recorded by the earlier run instead of silently dropping them.
        previous = json.loads(out.read_text(encoding="utf-8"))
        for key in ("zip_sha256", "zip_members", "zip_members_not_in_annotations", "zip_deleted"):
            if key not in record and key in previous:
                record[key] = previous[key]
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    v = record["validation"]
    print(f"{v['n_ok']}/{v['n_expected']} images OK; problems: "
          + ", ".join(f"{k}={len(p)}" for k, p in v["problems"].items()) + f"\nreport: {out}")
    return 0 if v["n_ok"] == v["n_expected"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
