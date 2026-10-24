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

DEPLOYABLE = ("full",)
ORACLE = ("oracle_gt", "oracle_source", "random_matched")
ALL_CONDITIONS = DEPLOYABLE + ORACLE


def vocabulary_for(condition: str, image_id: int, all_ids: list[int], gt_classes: Mapping[int, list[int]],
                   source_space: Mapping[int, set[int]], seed: int) -> list[int]:
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
