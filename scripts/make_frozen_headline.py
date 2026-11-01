"""Apply the FROZEN headline method (D34) to any split, with no recalibration.

D34 = RemoteCLIP global p_g (softmax of logit_scale x cosine over the 80 classes) >= t = 0.023681640625.
p_g is rebuilt from the plain global RemoteCLIP scores of `scripts/run_selection.py`; on dev this reproduces
outputs/selection/dev/m6_threshold_m8.npz exactly (checked below every time).

Also writes the D34 secondary methods with their dev-calibrated parameters (D32), frozen the same way:
threshold m = 20 (t = 0.004650115966796875) and top_p m = 20 (tau = 0.9464568403151777); each reproduces its dev mask
(outputs/selection/dev/m6_threshold_m20.npz, m6_top_p_m20.npz) before anything is written.

Usage:
    python scripts/make_frozen_headline.py --split test
    # writes outputs/selection/<split>/{headline_d34, frozen_threshold_m20, frozen_top_p_m20}.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aerial_ovd.vocabulary.retention import threshold, top_p  # noqa: E402
from aerial_ovd.vocabulary.selectors import softmax  # noqa: E402

D34_T = 0.023681640625
# D34 secondary methods: (output name, rule, parameter, dev mask it must reproduce)
SECONDARY = (("frozen_threshold_m20", threshold, 0.004650115966796875, "m6_threshold_m20"),
             ("frozen_top_p_m20", top_p, 0.9464568403151777, "m6_top_p_m20"))
SEL = REPO_ROOT / "outputs" / "selection"


def logit_scale() -> float:
    return float(json.loads(str(np.load(SEL / "dev" / "views_remoteclip_b32.npz")["meta"]))["logit_scale"]) \
        if (SEL / "dev" / "views_remoteclip_b32.npz").is_file() else 99.85401153564453  # RemoteCLIP ViT-B/32 exp(logit_scale)


def masks(scores: np.ndarray, ls: float, rule=threshold, param: float = D34_T) -> np.ndarray:
    return np.stack([rule(softmax(ls * s.astype(np.float64)), param) for s in scores])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True)
    args = ap.parse_args()
    ls = logit_scale()
    dev = np.load(SEL / "dev" / "remoteclip_b32.npz")
    assert (masks(dev["scores"], ls) == np.load(SEL / "dev" / "m6_threshold_m8.npz")["selected"]).all(), "D34 not reproduced on dev"
    z = np.load(SEL / args.split / "remoteclip_b32.npz")
    M = masks(z["scores"], ls)
    np.savez_compressed(SEL / args.split / "headline_d34.npz", image_ids=z["image_ids"], cat_ids=z["cat_ids"], selected=M,
                        meta=json.dumps({"selector": "headline_d34", "rule": "D34 frozen: RemoteCLIP global p_g >= t", "t": D34_T,
                                         "logit_scale": ls, "scores": f"outputs/selection/{args.split}/remoteclip_b32.npz",
                                         "scores_meta": json.loads(str(z["meta"]))}))
    print(f"{args.split} headline_d34: {len(M)} images, mean size {M.sum(1).mean():.2f} (min {M.sum(1).min()}, max {M.sum(1).max()})")
    for name, rule, param, dev_mask in SECONDARY:
        ref = np.load(SEL / "dev" / f"{dev_mask}.npz")["selected"]
        assert (masks(dev["scores"], ls, rule, param) == ref).all(), f"{dev_mask} not reproduced on dev"
        M = masks(z["scores"], ls, rule, param)
        np.savez_compressed(SEL / args.split / f"{name}.npz", image_ids=z["image_ids"], cat_ids=z["cat_ids"], selected=M,
                            meta=json.dumps({"selector": name, "rule": f"D34 secondary, frozen: {rule.__name__}({param})",
                                             "param": param, "dev_mask": f"outputs/selection/dev/{dev_mask}.npz",
                                             "logit_scale": ls, "scores": f"outputs/selection/{args.split}/remoteclip_b32.npz",
                                             "scores_meta": json.loads(str(z["meta"]))}))
        print(f"{args.split} {name}: mean size {M.sum(1).mean():.2f} (min {M.sum(1).min()}, max {M.sum(1).max()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
