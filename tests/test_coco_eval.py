"""Known-answer tests for the evaluator (M9): these must pass before any number is trusted."""

import pytest

from aerial_ovd.evaluation.coco_eval import evaluate, filter_source_aware


def _gt():
    # two images from different "sources": image 1 annotates class 1 only, image 2 class 2 only
    return {
        "images": [{"id": 1, "file_name": "A_1.jpg", "width": 100, "height": 100},
                   {"id": 2, "file_name": "B_2.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "small car"}, {"id": 2, "name": "vehicle"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [50, 50, 40, 40], "area": 1600, "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 2, "bbox": [5, 5, 60, 60], "area": 3600, "iscrowd": 0},
        ],
    }


SPACE = {1: {1}, 2: {2}}


def _perfect():
    return [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0}
            for a in _gt()["annotations"]]


@pytest.mark.parametrize("protocol", ["standard", "source_aware"])
def test_ground_truth_as_predictions_scores_one(protocol):
    r = evaluate(_gt(), _perfect(), [1, 2], protocol=protocol, image_space=SPACE)
    for m in (100, 300, 900):
        assert r[f"AP@{m}"] == pytest.approx(1.0)
        assert r[f"AP50@{m}"] == pytest.approx(1.0)
        assert r[f"AR@{m}"] == pytest.approx(1.0)
    assert r["n_classes_with_gt"] == 2


def test_cross_source_prediction_hurts_standard_but_not_source_aware():
    # a confident "vehicle" (class 2) box on image 1, where class 2 is never annotated
    preds = _perfect() + [{"image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20], "score": 1.0}]
    std = evaluate(_gt(), preds, [1, 2], protocol="standard", image_space=SPACE)
    src = evaluate(_gt(), preds, [1, 2], protocol="source_aware", image_space=SPACE)
    assert std["per_class"]["2"]["AP50"] < 1.0      # counted as a false positive
    assert src["per_class"]["2"]["AP50"] == pytest.approx(1.0)  # ignored: not annotated there
    assert src["n_predictions"] == std["n_predictions"] - 1


def test_wrong_boxes_score_zero_and_empty_predictions_are_handled():
    wrong = [{"image_id": 1, "category_id": 1, "bbox": [70, 0, 10, 10], "score": 0.9}]
    r = evaluate(_gt(), wrong, [1, 2], protocol="standard", image_space=SPACE)
    assert r["AP50@900"] == pytest.approx(0.0)
    e = evaluate(_gt(), [], [1, 2], protocol="standard")
    assert e["empty_predictions"] and e["AP50@900"] == 0.0


def test_failed_image_stays_in_denominator():
    # predictions only for image 1: class 2 (image 2) is entirely missed -> mean AP halves
    preds = [p for p in _perfect() if p["image_id"] == 1]
    r = evaluate(_gt(), preds, [1, 2], protocol="standard", image_space=SPACE)
    assert r["AP50@900"] == pytest.approx(0.5)


def test_maxdets_cap_limits_recall():
    # 150 GT boxes in one image; only top-100 counted at maxDets=100
    anns = [{"id": i + 1, "image_id": 1, "category_id": 1, "bbox": [i % 15 * 6, i // 15 * 9, 5, 8],
             "area": 40, "iscrowd": 0} for i in range(150)]
    gt = {"images": [{"id": 1, "file_name": "A_1.jpg", "width": 100, "height": 100}],
          "categories": [{"id": 1, "name": "small car"}], "annotations": anns}
    preds = [{"image_id": 1, "category_id": 1, "bbox": a["bbox"], "score": 1.0 - i * 1e-4} for i, a in enumerate(anns)]
    r = evaluate(gt, preds, [1], protocol="standard")
    assert r["AR@100"] == pytest.approx(100 / 150, abs=1e-3)
    assert r["AR@900"] == pytest.approx(1.0)


def test_filter_and_bad_inputs():
    assert filter_source_aware([{"image_id": 1, "category_id": 2}], SPACE) == []
    with pytest.raises(ValueError):
        evaluate(_gt(), [], [1, 99])
    with pytest.raises(ValueError):
        evaluate(_gt(), _perfect(), [1, 2], protocol="source_aware")


def _random_case(seed):
    import random

    rng = random.Random(seed)
    images = [{"id": i, "file_name": f"A_{i}.jpg", "width": 200, "height": 200} for i in range(1, 9)]
    cats = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 3, "name": "c"}]
    anns, preds, aid = [], [], 1
    for im in images:
        for _ in range(rng.randint(0, 6)):
            x, y = rng.uniform(0, 150), rng.uniform(0, 150)
            w, h = rng.uniform(4, 50), rng.uniform(4, 50)
            c = rng.choice([1, 2, 3])
            anns.append({"id": aid, "image_id": im["id"], "category_id": c, "bbox": [x, y, w, h],
                         "area": w * h, "iscrowd": 0})
            aid += 1
            if rng.random() < 0.7:  # a jittered true positive (sometimes the wrong class)
                preds.append({"image_id": im["id"], "category_id": c if rng.random() < 0.8 else rng.choice([1, 2, 3]),
                              "bbox": [x + rng.uniform(-6, 6), y + rng.uniform(-6, 6), w, h], "score": rng.random()})
        for _ in range(rng.randint(0, 4)):  # pure false positives
            preds.append({"image_id": im["id"], "category_id": rng.choice([1, 2, 3]),
                          "bbox": [rng.uniform(0, 150), rng.uniform(0, 150), 20, 20], "score": rng.random()})
    return {"images": images, "categories": cats, "annotations": anns}, preds


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_fast_ap50_reproduces_cocoeval(seed):
    from aerial_ovd.evaluation.coco_eval import ap50_from_matches, per_image_matches

    gt, preds = _random_case(seed)
    ids = [im["id"] for im in gt["images"]]
    ref = evaluate(gt, preds, ids, protocol="standard")["AP50@900"]
    fast = ap50_from_matches(per_image_matches(gt, preds, ids))
    assert fast == pytest.approx(ref, abs=1e-9)


def test_paired_bootstrap_identical_conditions_gives_zero():
    from aerial_ovd.evaluation.coco_eval import paired_bootstrap, per_image_matches

    gt, preds = _random_case(5)
    ids = [im["id"] for im in gt["images"]]
    m = per_image_matches(gt, preds, ids)
    r = paired_bootstrap(m, m, n_boot=50, seed=1)
    assert r["diff_point"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0 and not r["ci_excludes_zero"]


def test_paired_bootstrap_reports_bias_diagnostics():
    from aerial_ovd.evaluation.coco_eval import paired_bootstrap, per_image_matches

    gt, preds_a = _random_case(6)
    _, preds_b = _random_case(7)
    ids = [im["id"] for im in gt["images"]]
    r = paired_bootstrap(per_image_matches(gt, preds_a, ids), per_image_matches(gt, preds_b, ids), n_boot=100, seed=0)
    for key in ("boot_mean", "boot_sd", "bias", "bias_flag", "basic_ci_low", "basic_ci_high"):
        assert key in r
    assert r["basic_ci_low"] == pytest.approx(2 * r["diff_point"] - r["ci_high"])
    assert r["bias"] == pytest.approx(r["boot_mean"] - r["diff_point"])
