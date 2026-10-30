"""Vocabulary conditions for headroom experiments (E0) and simple baselines.

Two kinds, kept apart on purpose:
  * DEPLOYABLE: may only use the image pixels (rule D16). Here only `full` (it uses nothing).
  * ORACLE DIAGNOSTICS / CONTROLS: use ground-truth-derived information and are NEVER methods.
      - oracle_gt:      the classes annotated in this image
      - oracle_source:  the label space of the image's source (6-34 classes)
      - random_matched: random classes, as many as oracle_gt (uses the GT class COUNT, so it is a
                        size-matched control, not a deployable method)
All conditions return category ids sorted ascending, so prompt order is identical across conditions.
"""

from __future__ import annotations

import random
from typing import Mapping

DEPLOYABLE = ("full",)  # plus "sel:<selector>:k<k>" (precomputed deployable selector scores, top-k)
ORACLE = ("oracle_gt", "oracle_source", "random_matched")
ALL_CONDITIONS = DEPLOYABLE + ORACLE


def parse_selector_condition(condition: str) -> tuple[str, int]:
    """'sel:remoteclip_b32:k8' -> ('remoteclip_b32', 8)."""
    parts = condition.split(":")
    if len(parts) != 3 or parts[0] != "sel" or not parts[2].startswith("k") or not parts[2][1:].isdigit():
        raise ValueError(f"bad selector condition {condition!r}; expected 'sel:<selector>:k<int>'")
    k = int(parts[2][1:])
    if k < 1:
        raise ValueError("budget k must be >= 1")
    return parts[1], k


def vocabulary_for(condition: str, image_id: int, all_ids: list[int], gt_classes: Mapping[int, list[int]],
                   source_space: Mapping[int, set[int]], seed: int,
                   selector_scores: Mapping[str, Mapping[int, "object"]] | None = None) -> list[int]:
    """selector_scores: selector name -> image_id -> score vector aligned with sorted(all_ids).

    'set:<name>' conditions use a precomputed variable-size selection: selector_scores[name][image_id]
    is then a boolean mask aligned with sorted(all_ids)."""
    if condition.startswith("set:"):
        import numpy as np

        name = condition.split(":", 1)[1]
        if not name or selector_scores is None or name not in selector_scores:
            raise ValueError(f"no precomputed selection for {condition!r}")
        mask = np.asarray(selector_scores[name][image_id], dtype=bool)
        ids_sorted = sorted(all_ids)
        ids = [ids_sorted[i] for i in np.where(mask)[0]]
        if not ids:
            raise ValueError(f"empty selection for image {image_id} under {condition}")
        return ids
    if condition.startswith("sel:"):
        import numpy as np

        name, k = parse_selector_condition(condition)
        if selector_scores is None or name not in selector_scores:
            raise ValueError(f"no precomputed scores for selector {name!r}")
        s = np.asarray(selector_scores[name][image_id], dtype=float)
        if np.isnan(s).any():
            raise ValueError(f"selector {name} failed on image {image_id} (NaN scores)")
        order = np.argsort(-s, kind="stable")[: min(k, len(s))]
        ids_sorted = sorted(all_ids)
        return sorted(ids_sorted[i] for i in order)
    if condition == "full":
        return sorted(all_ids)
    if condition == "oracle_gt":
        ids = sorted(gt_classes[image_id])
    elif condition == "oracle_source":
        ids = sorted(source_space[image_id])
    elif condition == "random_matched":
        k = len(gt_classes[image_id])
        rng = random.Random(f"{seed}:{image_id}")  # per-image seed: stable and order-independent
        ids = sorted(rng.sample(sorted(all_ids), k))
    else:
        raise ValueError(f"unknown condition {condition!r}; known: {ALL_CONDITIONS}")
    if not ids:
        raise ValueError(f"empty vocabulary for image {image_id} under {condition}")
    return ids
