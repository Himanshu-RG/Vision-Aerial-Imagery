import pytest

from aerial_ovd.data.lae80c import build_manifest, clean_coco, source_label_space, source_of, subset_coco
from aerial_ovd.data.split import iterative_stratification
from aerial_ovd.vocabulary.names import build_prompt, prompt_name


def _coco():
    return {
        "images": [
            {"id": 1, "file_name": "FAIR-18C_1.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "DIOR-18C_2.jpg", "width": 50, "height": 50},
            {"id": 3, "file_name": "DIOR-18C_3.jpg", "width": 50, "height": 50},
        ],
        "categories": [{"id": 1, "name": "Small Car"}, {"id": 2, "name": "storagetank"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [90, 90, 20, 20], "area": 400, "iscrowd": 0},  # overflows
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [10, 10, 5, 5], "area": 25, "iscrowd": 0},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [10, 10, 5, 5], "area": 25, "iscrowd": 0},    # duplicate
            {"id": 4, "image_id": 2, "category_id": 2, "bbox": [0, 0, 0, 10], "area": 0, "iscrowd": 0},      # zero width
            {"id": 5, "image_id": 2, "category_id": 2, "bbox": [5, 5, 1, 30], "area": 30, "iscrowd": 0},     # tiny side
            {"id": 6, "image_id": 2, "category_id": 99, "bbox": [5, 5, 5, 5], "area": 25, "iscrowd": 0},     # bad cat
        ],
    }


def test_source_of():
    assert source_of("Xview-34C_2207_49_512_512_1536_1536.jpg") == "Xview-34C"
    with pytest.raises(ValueError):
        source_of("COCO_0001.jpg")


def test_clean_reports_every_change():
    clean, rep = clean_coco(_coco())
    c = rep.counts
    assert c["clipped_to_image"] == 1 and c["exact_duplicate"] == 1
    assert c["nonpositive_box"] == 1 and c["dropped_unknown_category"] == 1
    assert c["tiny_box_flagged"] == 1 and c["image_without_annotations"] == 1  # image 3
    assert rep.n_annotations_in == 6 and rep.n_annotations_out == 3
    clipped = next(a for a in clean["annotations"] if a["id"] == 1)
    assert clipped["bbox"] == [90.0, 90.0, 10.0, 10.0] and clipped["area"] == 100.0
    assert rep.clipped_min_area_kept == pytest.approx(0.25)


def test_clean_does_not_mutate_input():
    raw = _coco()
    clean_coco(raw)
    assert raw["annotations"][0]["bbox"] == [90, 90, 20, 20] and len(raw["annotations"]) == 6


def test_manifest_and_label_space():
    clean, _ = clean_coco(_coco())
    rows = {r["image_id"]: r for r in build_manifest(clean)}
    assert rows[1]["n_boxes"] == 2 and rows[1]["classes"] == "1" and rows[1]["n_small"] == 2
    assert rows[3]["n_boxes"] == 0 and rows[3]["classes"] == ""
    assert source_label_space(clean) == {"FAIR-18C": {1}, "DIOR-18C": {2}}
    sub = subset_coco(clean, {2})
    assert {a["image_id"] for a in sub["annotations"]} == {2} and len(sub["categories"]) == 2


def test_split_is_deterministic_disjoint_complete_and_stratified():
    labels = {i: ([i % 7] + ([99] if i % 25 == 0 else [])) for i in range(500)}
    a = iterative_stratification(labels, [0.2, 0.8], seed=3)
    b = iterative_stratification(labels, [0.2, 0.8], seed=3)
    assert a == b
    assert set(a[0]).isdisjoint(a[1]) and set(a[0]) | set(a[1]) == set(labels)
    assert abs(len(a[0]) - 100) <= 2
    rare_dev = sum(1 for i in a[0] if 99 in labels[i])  # 20 items carry the rare label
    assert rare_dev == 4


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        iterative_stratification({1: [1]}, [0.5, 0.6], seed=0)


def test_prompt_names():
    assert prompt_name("groundtrackfield") == "ground track field"
    assert prompt_name("Front loader/Bulldozer") == "front loader/bulldozer"
    assert prompt_name("Small Car") == "small car"
    assert prompt_name("Small Car", policy="raw") == "Small Car"
    assert build_prompt(["a", "b c"]) == "a. b c."
    with pytest.raises(ValueError):
        build_prompt([])
