"""Apply the FROZEN headline method (D34) to any split, with no recalibration.

D34 = RemoteCLIP global p_g (softmax of logit_scale x cosine over the 80 classes) >= t = 0.023681640625.
p_g is rebuilt from the plain global RemoteCLIP scores of `scripts/run_selection.py`; on dev this reproduces
outputs/selection/dev/m6_threshold_m8.npz exactly (checked below every time).

Usage:
    python scripts/make_frozen_headline.py --split nwpu      # writes outputs/selection/<split>/headline_d34.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aerial_ovd.vocabulary.retention import threshold  # noqa: E402
from aerial_ovd.vocabulary.selectors import softmax  # noqa: E402

D34_T = 0.023681640625
SEL = REPO_ROOT / "outputs" / "selection"


def logit_scale() -> float:
    return float(json.loads(str(np.load(SEL / "dev" / "views_remoteclip_b32.npz")["meta"]))["logit_scale"]) \
        if (SEL / "dev" / "views_remoteclip_b32.npz").is_file() else 99.85401153564453  # RemoteCLIP ViT-B/32 exp(logit_scale)


def masks(scores: np.ndarray, ls: float) -> np.ndarray:
    return np.stack([threshold(softmax(ls * s.astype(np.float64)), D34_T) for s in scores])


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
    print(f"{args.split}: {len(M)} images, mean size {M.sum(1).mean():.2f} (min {M.sum(1).min()}, max {M.sum(1).max()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
