import numpy as np

from aerial_ovd.evaluation.failure import density_bin, det_causes, gt_outcomes, iou_xywh, size_bin

GT = np.array([[0, 0, 10, 10], [100, 100, 10, 10], [200, 200, 10, 10], [300, 300, 10, 10], [400, 400, 10, 10]])
GC = [1, 1, 2, 3, 4]


def test_iou_known_values():
    assert np.isclose(iou_xywh([[0, 0, 10, 10]], [[0, 0, 10, 10]])[0, 0], 1.0)
    assert np.isclose(iou_xywh([[0, 0, 10, 10]], [[5, 0, 10, 10]])[0, 0], 50 / 150)
    assert iou_xywh([[0, 0, 10, 10]], [[20, 20, 5, 5]])[0, 0] == 0.0


def test_gt_outcomes_every_case():
    det = np.array([[0, 0, 10, 10],        # detects GT0 (class 1)
                    [100, 100, 10, 10],    # class 9 on GT1 -> wrong_class
                    [205, 200, 10, 10],    # class 2 on GT2 at IoU 1/3 -> poor_box
                    [400, 400, 10, 10]])   # class 7 on GT4, whose class 4 is not in the vocabulary
    out = gt_outcomes(GT, GC, det, [1, 9, 2, 7], [0.9, 0.8, 0.7, 0.6], vocab={1, 2, 3, 9, 7})
    assert out == [("detected", None), ("wrong_class", 9), ("poor_box", None), ("missed", None), ("vocab_miss", 7)]


def test_gt_outcomes_threshold_drops_low_scores():
    out = gt_outcomes(GT[:1], [1], [[0, 0, 10, 10]], [1], [0.1], vocab={1}, thr=0.14)
    assert out == [("missed", None)]


def test_det_causes_every_case():
    det = np.array([[0, 0, 10, 10],        # tp (GT0, class 1)
                    [0, 0, 10, 10],        # duplicate of GT0
                    [100, 100, 10, 10],    # class 2 on GT1 (class 1) -> wrong_class
                    [205, 200, 10, 10],    # class 2, IoU 1/3 with GT2 -> poor_box
                    [600, 600, 10, 10],    # class 3 occurs in the image -> bg_present
                    [700, 700, 10, 10],    # class 8 does not occur -> bg_absent
                    [800, 800, 10, 10]])   # below threshold -> ignored
    causes, idx = det_causes(GT, GC, det, [1, 1, 2, 2, 3, 8, 1], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.01], thr=0.14)
    assert causes == ["tp", "duplicate", "wrong_class", "poor_box", "bg_present", "bg_absent"]
    assert list(idx) == [0, 1, 2, 3, 4, 5]


def test_bins():
    assert [size_bin(10, 10), size_bin(20, 20), size_bin(50, 50), size_bin(100, 100)] == ["tiny", "small", "medium", "large"]
    assert [density_bin(n) for n in (1, 6, 21, 101)] == ["1-5", "6-20", "21-100", ">100"]
