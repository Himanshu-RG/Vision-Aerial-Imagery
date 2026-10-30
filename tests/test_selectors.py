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


def test_crop_boxes_cover_image_with_overlap():
    from aerial_ovd.vocabulary.selectors import crop_boxes

    b = crop_boxes(600, 600, 3, 0.25)
    assert len(b) == 9
    assert min(x0 for x0, _, _, _ in b) == 0 and max(x1 for _, _, x1, _ in b) == 600
    assert min(y0 for _, y0, _, _ in b) == 0 and max(y1 for _, _, _, y1 in b) == 600
    assert all(x1 - x0 == 250 and y1 - y0 == 250 for x0, y0, x1, y1 in b)   # 600/3*1.25
    assert crop_boxes(100, 50, 1) == [(0, 0, 100, 50)]
    with pytest.raises(ValueError):
        crop_boxes(100, 100, 0)


def test_combine_views_limits_and_aggregation():
    from aerial_ovd.vocabulary.selectors import combine_views, softmax

    g = np.array([2.0, 0.0, 0.0])
    crops = np.array([[0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [5.0, 0.0, 0.0]])
    gids = np.array([2, 2, 3])
    assert np.allclose(combine_views(g, crops, gids, 1.0), softmax(g))
    local_max = combine_views(g, crops, gids, 0.0, "max", grids=(2,))
    assert np.allclose(local_max, softmax(crops[:2], axis=-1).max(0))     # grid-3 crop excluded
    local_mean = combine_views(g, crops, gids, 0.0, "mean", grids=(2, 3))
    assert np.allclose(local_mean, softmax(crops, axis=-1).mean(0))
    mix = combine_views(g, crops, gids, 0.5, "max", grids=(2, 3))
    assert np.allclose(mix, 0.5 * softmax(g) + 0.5 * softmax(crops, axis=-1).max(0))
    with pytest.raises(ValueError):
        combine_views(g, crops, gids, 1.5)


def test_set_condition_uses_boolean_mask():
    from aerial_ovd.vocabulary.conditions import vocabulary_for

    all_ids = [10, 2, 3]  # sorted -> [2, 3, 10]
    sel = {"m6": {5: np.array([True, False, True])}}
    assert vocabulary_for("set:m6", 5, all_ids, {}, {}, 0, sel) == [2, 10]
    with pytest.raises(ValueError):
        vocabulary_for("set:m6", 5, all_ids, {}, {}, 0, {"m6": {5: np.array([False, False, False])}})
    with pytest.raises(ValueError):
        vocabulary_for("set:nope", 5, all_ids, {}, {}, 0, sel)
