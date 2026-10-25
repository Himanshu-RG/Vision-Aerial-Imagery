import inspect

import numpy as np
import pytest
from PIL import Image

from aerial_ovd.vocabulary.selectors import (
    CLIPSelector, FrequencySelector, RandomSelector, Selector, vocabulary_recall,
)


def _img(seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))


def test_top_k_is_deterministic_with_ties():
    s = np.array([0.5, 0.9, 0.5, 0.1])
    assert Selector.top_k(s, 1).tolist() == [1]
    assert Selector.top_k(s, 2).tolist() == [0, 1]       # tie 0 vs 2 -> lower index wins
    assert Selector.top_k(s, 10).tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        Selector.top_k(s, 0)


def test_frequency_selector_is_image_independent():
    f = FrequencySelector(np.array([3, 1, 2]))
    assert f.scores(_img(0)).tolist() == f.scores(_img(1)).tolist() == [3, 1, 2]


def test_random_selector_is_seeded_and_content_based():
    r = RandomSelector(10, seed=0)
    assert np.array_equal(r.scores(_img(0)), r.scores(_img(0)))
    assert not np.array_equal(r.scores(_img(0)), r.scores(_img(1)))
    assert not np.array_equal(r.scores(_img(0)), RandomSelector(10, seed=1).scores(_img(0)))


def test_selector_api_only_accepts_the_image():
    # Rule D16: selectors must not receive file names, sources, sizes or annotations.
    for cls in (FrequencySelector, RandomSelector, CLIPSelector):
        params = list(inspect.signature(cls.scores).parameters)
        assert params == ["self", "image"], (cls.__name__, params)


def test_vocabulary_recall():
    assert vocabulary_recall(np.array([1, 2, 3]), {2, 9}) == 0.5
    with pytest.raises(ValueError):
        vocabulary_recall(np.array([1]), set())


def test_selector_condition_parsing_and_topk_vocabulary():
    from aerial_ovd.vocabulary.conditions import parse_selector_condition, vocabulary_for

    assert parse_selector_condition("sel:remoteclip_b32:k8") == ("remoteclip_b32", 8)
    for bad in ("sel:x", "sel:x:8", "sel:x:k0", "clip:x:k3"):
        with pytest.raises(ValueError):
            parse_selector_condition(bad)
    all_ids = [3, 1, 2, 10]  # any order in; scores are aligned with sorted ids [1, 2, 3, 10]
    scores = {"s": {7: np.array([0.1, 0.9, 0.5, 0.7])}}
    assert vocabulary_for("sel:s:k2", 7, all_ids, {}, {}, 0, scores) == [2, 10]
    assert vocabulary_for("sel:s:k99", 7, all_ids, {}, {}, 0, scores) == [1, 2, 3, 10]
    with pytest.raises(ValueError):
        vocabulary_for("sel:missing:k2", 7, all_ids, {}, {}, 0, scores)
    with pytest.raises(ValueError):
        vocabulary_for("sel:s:k2", 7, all_ids, {}, {}, 0, {"s": {7: np.array([np.nan, 1, 1, 1])}})
