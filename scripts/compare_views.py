"""Check that views computed elsewhere (e.g. GPU, clean commit) reproduce the committed M5 selectors.

Usage:
    python scripts/compare_views.py outputs/selection_gpu/dev/views_remoteclip_b32.npz outputs/selection/dev/remoteclip_b32_gl.npz
Recomputes the committed selector's variant (alpha, agg, grids from its meta) from the given views and
reports top-k agreement (k = 3, 8, 20), max |score diff|, and median ms/img of the given views file.
The views file must contain at least the crop grids the variant uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aerial_ovd.vocabulary.selectors import combine_views  # noqa: E402


def main() -> int:
    views, ref = Path(sys.argv[1]), Path(sys.argv[2])
    v, r = np.load(views), np.load(ref)
    rm = json.loads(str(r["meta"]))
    vm = json.loads(str(v["meta"]))
    if list(v["image_ids"]) != list(r["image_ids"]):
        print("different image sets", file=sys.stderr)
        return 1
    S = np.stack([combine_views(v["global_logits"][i], v["crop_logits"][i], v["crop_grid_ids"], rm["alpha"],
                                rm["agg"], tuple(rm["grids"])) for i in range(len(v["image_ids"]))])
    out = {"selector": rm["selector"], "variant": rm["variant"], "views_device": vm["device"], "views_host": vm["host"],
           "views_git_commit": vm["git_commit"], "views_git_dirty_code": vm["git_dirty_code"],
           "views_grids": vm["grids"], "max_abs_score_diff": float(np.abs(S - r["scores"]).max()),
           "views_ms_median": float(np.nanmedian(v["image_ms"]))}
    for k in (3, 8, 20):
        a = np.argsort(-S, axis=1, kind="stable")[:, :k]
        b = np.argsort(-r["scores"], axis=1, kind="stable")[:, :k]
        out[f"k{k}_identical_sets"] = float(np.mean([set(x) == set(y) for x, y in zip(a, b)]))
    print(json.dumps(out, indent=2))
    dst = views.parent / f"comparison_{rm['selector']}.json"
    dst.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
