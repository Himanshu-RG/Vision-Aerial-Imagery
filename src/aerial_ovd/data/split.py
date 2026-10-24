"""Deterministic multi-label iterative stratification (Sechidis, Tsoumakas & Vlahavas, ECML-PKDD 2011).

Used to split images so that every class keeps (as far as possible) the same dev/test ratio,
which keeps rare classes represented in the small dev split.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Hashable, Mapping, Sequence


def iterative_stratification(
    labels: Mapping[Hashable, Sequence[Hashable]],
    fractions: Sequence[float],
    seed: int,
) -> list[list[Hashable]]:
    """Split item ids into len(fractions) folds.

    labels: item id -> labels present on that item.
    Returns one sorted list of item ids per fold. Deterministic for a given seed and input.
    """
    if not fractions or any(f < 0 for f in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must be non-negative and sum to 1")
    rng = random.Random(seed)
    items = sorted(labels, key=repr)
    n_folds = len(fractions)

    desired_total = [f * len(items) for f in fractions]
    label_items: dict[Hashable, list] = defaultdict(list)
    for it in items:
        for lab in set(labels[it]):
            label_items[lab].append(it)
    desired_label = {lab: [f * len(its) for f in fractions] for lab, its in label_items.items()}

    assigned: dict[Hashable, int] = {}
    folds: list[list] = [[] for _ in range(n_folds)]

    def assign(it, fold):
        assigned[it] = fold
        folds[fold].append(it)
        desired_total[fold] -= 1
        for lab in set(labels[it]):
            desired_label[lab][fold] -= 1

    while True:
        remaining = {lab: [it for it in its if it not in assigned] for lab, its in label_items.items()}
        remaining = {lab: its for lab, its in remaining.items() if its}
        if not remaining:
            break
        # Rarest remaining label first; ties broken by label repr for determinism.
        lab = min(remaining, key=lambda lb: (len(remaining[lb]), repr(lb)))
        pool = remaining[lab]
        rng.shuffle(pool)
        for it in pool:
            want = desired_label[lab]
            best = max(want)
            cands = [k for k in range(n_folds) if want[k] == best]
            if len(cands) > 1:
                best_tot = max(desired_total[k] for k in cands)
                cands = [k for k in cands if desired_total[k] == best_tot]
            assign(it, rng.choice(cands) if len(cands) > 1 else cands[0])

    # Items without any label: fill the folds that are furthest below their target size.
    for it in items:
        if it not in assigned:
            best_tot = max(desired_total)
            cands = [k for k in range(n_folds) if desired_total[k] == best_tot]
            assign(it, rng.choice(cands) if len(cands) > 1 else cands[0])

    return [sorted(f, key=repr) for f in folds]
