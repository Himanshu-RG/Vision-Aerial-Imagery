"""Check that selections computed elsewhere (e.g. on the GPU) match the committed ones, and report timing.

Usage:
    python scripts/compare_selection.py outputs/selection/dev outputs/selection_gpu/dev
For each selector present in both folders: top-k agreement at k in {3, 8, 20}, max |score diff|, and
median selector time per image in each folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ref", type=Path)
    ap.add_argument("other", type=Path)
    args = ap.parse_args()
    out = {}
    for f in sorted(args.other.glob("*.npz")):
        r = args.ref / f.name
        if not r.is_file():
            continue
        a, b = np.load(r), np.load(f)
        if list(a["image_ids"]) != list(b["image_ids"]):
            print(f"{f.stem}: different image sets", file=sys.stderr)
            continue
        sa, sb = a["scores"], b["scores"]
        agree = {}
        for k in (3, 8, 20):
            ta = np.argsort(-sa, axis=1, kind="stable")[:, :k]
            tb = np.argsort(-sb, axis=1, kind="stable")[:, :k]
            agree[f"k{k}_identical_sets"] = float(np.mean([set(x) == set(y) for x, y in zip(ta, tb)]))
        out[f.stem] = {"max_abs_score_diff": float(np.nanmax(np.abs(sa - sb))), **agree,
                       "ref_ms_median": float(np.median(a["image_ms"])), "other_ms_median": float(np.median(b["image_ms"])),
                       "ref_device": json.loads(str(a["meta"]))["device"], "other_device": json.loads(str(b["meta"]))["device"],
                       "other_host": json.loads(str(b["meta"]))["host"]}
    print(json.dumps(out, indent=2))
    (args.other / "comparison_vs_committed.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
