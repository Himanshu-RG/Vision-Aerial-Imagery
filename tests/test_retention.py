import numpy as np
import pytest

from aerial_ovd.vocabulary.retention import calibrate, retain, threshold, top_p, topk

P = np.array([0.5, 0.2, 0.15, 0.1, 0.05])


def test_topk_and_never_empty():
    assert topk(P, 2).tolist() == [True, True, False, False, False]
    assert topk(P, 0).sum() == 1
    assert threshold(P, 0.99).tolist() == [True, False, False, False, False]  # falls back to top-1


def test_threshold_and_cap():
    assert threshold(P, 0.12).tolist() == [True, True, True, False, False]
    assert threshold(P, 0.0, max_k=2).tolist() == [True, True, False, False, False]


def test_top_p():
    assert top_p(P, 0.5).sum() == 1
    assert top_p(P, 0.6).sum() == 2           # 0.5 + 0.2 >= 0.6
    assert top_p(P, 0.85).sum() == 3
    assert top_p(P, 1.0).sum() == 5


def test_retain_adds_locally_supported_classes():
    pl = np.array([0.1, 0.1, 0.1, 0.6, 0.1])  # class 3 strong locally, weak globally
    m = retain(P, pl, 1, 0.3)
    assert m.tolist() == [True, False, False, True, False]
    assert retain(P, pl, 1, 0.9).sum() == 1


def test_calibrate_hits_target_mean_size():
    rng = np.random.default_rng(0)
    probs = [rng.dirichlet(np.ones(20)) for _ in range(200)]
    size = lambda t: np.mean([threshold(p, t).sum() for p in probs])
    t = calibrate(size, 5.0, 0.0, 1.0, increasing=False)
    assert size(t) == pytest.approx(5.0, abs=0.2)
