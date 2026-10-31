from pathlib import Path

import pytest
import torch

from aerial_ovd.detector.gdino import build_caption, positive_map_from_spans
from aerial_ovd.vocabulary.conditions import ALL_CONDITIONS, vocabulary_for


def test_caption_matches_official_format():
    cap, spans = build_caption(["Small Car", "storage tank"])
    assert cap == "small car . storage tank ."
    assert [cap[a:b] for a, b in spans] == ["small car", "storage tank"]
    with pytest.raises(ValueError):
        build_caption([])


def _tokenizer():
    from aerial_ovd.utils.paths import StoreNotConfigured, get_store

    try:
        d = get_store() / "checkpoints" / "grounding-dino-tiny"
    except StoreNotConfigured:
        pytest.skip("store not configured")
    if not (d / "tokenizer.json").is_file():
        pytest.skip("grounding-dino-tiny tokenizer not downloaded")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(d)


def test_positive_map_covers_exactly_each_name():
    tok = _tokenizer()
    cap, spans = build_caption(["small car", "front loader/bulldozer", "dam"])
    enc = tok(cap)
    pmap = positive_map_from_spans(enc, spans)
    tokens = enc.tokens()
    for row, name in zip(pmap, ["small car", "front loader/bulldozer", "dam"]):
        idx = torch.nonzero(row).flatten().tolist()
        assert row.sum().item() == pytest.approx(1.0, abs=1e-4)
        covered = tok.convert_tokens_to_string([tokens[i] for i in idx]).replace(" ", "")
        assert covered == name.replace(" ", "")
        assert "." not in [tokens[i] for i in idx] and "[CLS]" not in [tokens[i] for i in idx]


def test_conditions_are_sorted_seeded_and_oracle_labelled():
    all_ids = list(range(1, 11))
    gt = {7: [5, 2]}
    space = {7: {1, 2, 3, 4, 5}}
    assert vocabulary_for("full", 7, all_ids, gt, space, 0) == all_ids
    assert vocabulary_for("oracle_gt", 7, all_ids, gt, space, 0) == [2, 5]
    assert vocabulary_for("oracle_source", 7, all_ids, gt, space, 0) == [1, 2, 3, 4, 5]
    r1 = vocabulary_for("random_matched", 7, all_ids, gt, space, 0)
    assert r1 == vocabulary_for("random_matched", 7, all_ids, gt, space, 0) and len(r1) == 2 and r1 == sorted(r1)
    assert set(ALL_CONDITIONS) == {"full", "oracle_gt", "oracle_source", "random_matched"}
    with pytest.raises(ValueError):
        vocabulary_for("clip_global", 7, all_ids, gt, space, 0)


def test_sample_images_is_exact_and_deterministic():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_detection import load_manifest, sample_images

    rows = load_manifest()
    for n in (4, 37, 200):
        a = sample_images(rows, "dev", n, seed=2026)
        assert len(a) == n and a == sample_images(rows, "dev", n, seed=2026)
        dev = {int(r["image_id"]) for r in rows if r["split"] == "dev"}
        assert set(a) <= dev  # never touches test
    assert len(sample_images(rows, "dev", None, seed=2026)) == len(dev)


def test_merge_detections_classwise_nms_and_topk():
    import numpy as np

    from aerial_ovd.detector.gdino import merge_detections

    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    classes = np.array([0, 0, 1, 0])  # box 1 duplicates box 0 (same class); box 2 is another class
    b, s, c = merge_detections(boxes, scores, classes, iou=0.5, top_k=10)
    assert s.tolist() == [0.9, 0.7, 0.6] and c.tolist() == [0, 1, 0]
    b, s, c = merge_detections(boxes, scores, classes, iou=0.5, top_k=2)
    assert s.tolist() == [0.9, 0.7]
