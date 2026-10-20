"""Download pinned checkpoints into <store>/checkpoints and record their exact commit + sha256.

Usage:
    python scripts/fetch_checkpoints.py --dry-run            # resolve commits + sizes, check disk, no download
    python scripts/fetch_checkpoints.py                      # download everything in configs/checkpoints.yaml
    python scripts/fetch_checkpoints.py --only grounding-dino-tiny

The lock file configs/checkpoints.lock.json is committed. Once an entry is locked, later
downloads use the locked commit, and the sha256 of every file must match the lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aerial_ovd.utils.config import load_yaml  # noqa: E402
from aerial_ovd.utils.paths import get_store  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "checkpoints.yaml"
LOCK = REPO_ROOT / "configs" / "checkpoints.lock.json"
DISK_MARGIN_GB = 1.0  # keep at least this much free after downloading


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_lock() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.is_file() else {}


def resolve(entry: dict, lock: dict) -> dict:
    """Ask the Hub for the commit SHA and file sizes (no download)."""
    from huggingface_hub import HfApi

    locked = lock.get(entry["name"], {})
    revision = locked.get("commit") or entry.get("revision") or "main"
    info = HfApi().model_info(entry["repo_id"], revision=revision, files_metadata=True)
    sizes = {s.rfilename: s.size for s in info.siblings}
    missing = [f for f in entry["files"] if f not in sizes]
    if missing:
        raise FileNotFoundError(f"{entry['repo_id']}@{info.sha}: missing files {missing}")
    return {"commit": info.sha, "sizes": {f: sizes[f] for f in entry["files"]}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="checkpoint names to process")
    args = parser.parse_args()

    entries = load_yaml(CONFIG)["checkpoints"]
    if args.only:
        unknown = set(args.only) - {e["name"] for e in entries}
        if unknown:
            print(f"unknown checkpoint names: {sorted(unknown)}", file=sys.stderr)
            return 2
        entries = [e for e in entries if e["name"] in args.only]

    store = get_store()
    # Keep the HF cache off the (small) system drive. Must be set before huggingface_hub is imported.
    os.environ.setdefault("HF_HOME", str(store / "hf_cache"))
    ckpt_root = store / "checkpoints"
    lock = load_lock()

    plans = []
    for e in entries:
        r = resolve(e, lock)
        target = ckpt_root / e["name"]
        todo = {f: s for f, s in r["sizes"].items() if not (target / f).is_file()}
        plans.append((e, r, target, todo))
        total_mb = sum(r["sizes"].values()) / 1024**2
        todo_mb = sum(todo.values()) / 1024**2
        print(f"{e['name']:<24} {e['repo_id']}@{r['commit'][:10]}  total {total_mb:8.1f} MB  to download {todo_mb:8.1f} MB")

    need_gb = sum(sum(t.values()) for *_, t in plans) / 1024**3
    free_gb = shutil.disk_usage(store).free / 1024**3
    print(f"\nstore: {store}\nneed {need_gb:.2f} GB, free {free_gb:.2f} GB (margin {DISK_MARGIN_GB} GB)")
    if need_gb + DISK_MARGIN_GB > free_gb:
        print("NOT ENOUGH DISK SPACE -- aborting.", file=sys.stderr)
        return 1
    if args.dry_run:
        print("dry run: nothing downloaded, lock file not changed.")
        return 0

    from huggingface_hub import hf_hub_download

    for e, r, target, _ in plans:
        record = {"repo_id": e["repo_id"], "commit": r["commit"], "files": {},
                  "fetched_at": datetime.now(timezone.utc).isoformat()}
        for fname, size in r["sizes"].items():
            path = Path(hf_hub_download(e["repo_id"], fname, revision=r["commit"], local_dir=target))
            digest = sha256_file(path)
            old = lock.get(e["name"], {}).get("files", {}).get(fname, {}).get("sha256")
            if old and old != digest:
                print(f"SHA256 MISMATCH for {e['name']}/{fname}: lock {old} vs file {digest}", file=sys.stderr)
                return 3
            record["files"][fname] = {"size": size, "sha256": digest}
            print(f"  ok {e['name']}/{fname}  sha256={digest[:16]}...")
        lock[e["name"]] = record
        LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    print(f"lock written: {LOCK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
