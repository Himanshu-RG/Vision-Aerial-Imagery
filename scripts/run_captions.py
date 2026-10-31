"""E5 (VocAda-style baseline), step 1: caption every image of a split with Florence-2-base.

Usage:
    python scripts/run_captions.py --split dev [--device cuda] [--limit N] [--out FILE]
Writes outputs/selection/<split>/captions_florence2_base.json:
    {"meta": {...}, "captions": {image_id: caption}, "ms": {image_id: caption time in ms}}
Generation is fixed a priori (D38): task <MORE_DETAILED_CAPTION>, num_beams=3, max_new_tokens=128, no sampling.
Only the decoded pixels are used (rule D16). CPU timings are not latency results (D9).
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aerial_ovd.utils.paths import get_store  # noqa: E402
from run_detection import load_manifest  # noqa: E402

TASK = "<MORE_DETAILED_CAPTION>"
GEN = {"num_beams": 3, "max_new_tokens": 128, "do_sample": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    store = get_store()
    ck = store / "checkpoints" / "florence2-base"
    processor = AutoProcessor.from_pretrained(ck)
    model = Florence2ForConditionalGeneration.from_pretrained(ck, dtype=torch.float32).to(args.device).eval()
    rows = sorted((r for r in load_manifest() if r["split"] == args.split), key=lambda r: int(r["image_id"]))
    if args.limit:
        rows = rows[: args.limit]
    out_path = args.out or (REPO_ROOT / "outputs" / "selection" / args.split / "captions_florence2_base.json")
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else {"captions": {}, "ms": {}}
    caps, ms, failed = done["captions"], done["ms"], []
    for i, r in enumerate(rows):
        iid = str(int(r["image_id"]))
        if iid in caps:
            continue
        try:
            with Image.open(store / "datasets" / "LAE-80C-images" / r["file_name"]) as im:
                image = im.convert("RGB")
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t = time.perf_counter()
            inputs = processor(text=TASK, images=image, return_tensors="pt").to(args.device)
            with torch.no_grad():
                ids = model.generate(**inputs, **GEN)
            text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            ms[iid] = round((time.perf_counter() - t) * 1e3, 1)
            caps[iid] = text
        except Exception as e:  # noqa: BLE001 - recorded; image keeps an empty caption -> falls back to top-1 later
            failed.append({"image_id": int(iid), "error": repr(e)})
        if (i + 1) % 50 == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"meta": {}, "captions": caps, "ms": ms}, indent=1), encoding="utf-8")
            print(f"  {i + 1}/{len(rows)}", flush=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs"],
                                capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip())
    meta = {"model": "florence-community/Florence-2-base", "task": TASK, "generation": GEN, "device": args.device,
            "host": socket.gethostname(), "git_commit": git, "git_dirty_code": dirty, "split": args.split,
            "n_images": len(rows), "failed": failed}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"meta": meta, "captions": caps, "ms": ms}, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")
    import statistics

    print(f"captions={len(caps)} failed={len(failed)} median_ms={statistics.median(ms.values()):.0f} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
