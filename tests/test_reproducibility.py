"""M13: reproducibility checks that need only committed files (no store, no models)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_frozen_split_matches_d18():
    import verify_artifacts

    assert verify_artifacts.check_split() == []


def test_frozen_masks_reproduce_from_committed_scores():
    import verify_artifacts

    assert verify_artifacts.check_masks() == []


def test_m12_median_ci_brackets_the_median():
    import m12_latency

    x = np.array([10.0, 11.0, 12.0, 13.0, 100.0])
    med, lo, hi = m12_latency.median_ci(x, n_boot=500)
    assert med == 12.0 and lo <= med <= hi
