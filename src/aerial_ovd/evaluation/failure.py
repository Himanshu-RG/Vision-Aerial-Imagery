"""M10 failure analysis: per-object outcomes and per-detection error causes (a diagnostic, not a metric).

Ground-truth side (every GT box, given one condition's vocabulary and detections with score >= thr):
  vocab_miss    the GT class was not in the image's selected vocabulary (a selection error)
  detected      a detection of the GT class overlaps it with IoU >= 0.5
  wrong_class   no such detection, but a detection of another class overlaps it with IoU >= 0.5
  poor_box      no IoU >= 0.5 detection at all, but one of the GT class overlaps with 0.1 <= IoU < 0.5
  missed        nothing of the GT class or any class at IoU >= 0.5, nothing of the GT class at IoU >= 0.1
"detected" is coverage (any detection), not COCO's one-to-one matching, so it is an upper bound on recall
at that threshold. For vocab_miss and wrong_class the class of the highest-scoring overlapping detection is
kept as `confused_as`.

Detection side (detections with score >= thr, processed by descending score, greedy one-to-one like COCO):
  tp            IoU >= 0.5 with a still-unclaimed GT of the same class
  duplicate     IoU >= 0.5 with a same-class GT that is already claimed
  wrong_class   IoU >= 0.5 with a GT of another class
  poor_box      0.1 <= IoU < 0.5 with a same-class GT
  bg_present    no overlap as above; the predicted class does occur in the image
  bg_absent     no overlap as above; the predicted class does not occur in the image (vocabulary can remove these)
"""

from __future__ import annotations

import numpy as np

GT_OUTCOMES = ("detected", "wrong_class", "poor_box", "missed", "vocab_miss")
DET_CAUSES = ("tp", "duplicate", "wrong_class", "poor_box", "bg_present", "bg_absent")
IOU_MATCH, IOU_LOC = 0.5, 0.1


def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) IoU matrix of boxes in COCO xywh format."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def gt_outcomes(gt_boxes, gt_cls, det_boxes, det_cls, det_scores, vocab, thr: float = 0.0) -> list[tuple[str, int | None]]:
    """One (outcome, confused_as) per GT box. Detections below `thr` are ignored."""
    gt_cls = np.asarray(gt_cls, dtype=int)
    det_cls, det_scores = np.asarray(det_cls, dtype=int), np.asarray(det_scores, dtype=float)
    keep = det_scores >= thr
    det_boxes, det_cls, det_scores = np.asarray(det_boxes, float).reshape(-1, 4)[keep], det_cls[keep], det_scores[keep]
    ious = iou_xywh(gt_boxes, det_boxes) if len(gt_cls) and len(det_cls) else np.zeros((len(gt_cls), 0))
    vocab = set(int(v) for v in vocab)
    out = []
    for g, c in enumerate(gt_cls):
        row = ious[g]
        same = det_cls == c
        other_hit = (~same) & (row >= IOU_MATCH)
        confused = int(det_cls[other_hit][np.argmax(det_scores[other_hit])]) if other_hit.any() else None
        if c not in vocab:
            out.append(("vocab_miss", confused))
        elif (same & (row >= IOU_MATCH)).any():
            out.append(("detected", None))
        elif confused is not None:
            out.append(("wrong_class", confused))
        elif (same & (row >= IOU_LOC)).any():
            out.append(("poor_box", None))
        else:
            out.append(("missed", None))
    return out


def det_causes(gt_boxes, gt_cls, det_boxes, det_cls, det_scores, thr: float) -> tuple[list[str], np.ndarray]:
    """Cause of every detection with score >= thr; returns (causes, indices of those detections)."""
    gt_cls = np.asarray(gt_cls, dtype=int)
    det_cls, det_scores = np.asarray(det_cls, dtype=int), np.asarray(det_scores, dtype=float)
    idx = np.where(det_scores >= thr)[0]
    idx = idx[np.argsort(-det_scores[idx], kind="mergesort")]
    db = np.asarray(det_boxes, float).reshape(-1, 4)
    ious = iou_xywh(db[idx], gt_boxes) if len(idx) and len(gt_cls) else np.zeros((len(idx), len(gt_cls)))
    present = set(gt_cls.tolist())
    claimed = np.zeros(len(gt_cls), bool)
    causes = []
    for r, d in enumerate(idx):
        row, k = ious[r], det_cls[d]
        same = gt_cls == k
        free = np.where(same & ~claimed & (row >= IOU_MATCH))[0]
        if len(free):
            claimed[free[np.argmax(row[free])]] = True
            causes.append("tp")
        elif (same & (row >= IOU_MATCH)).any():
            causes.append("duplicate")
        elif ((~same) & (row >= IOU_MATCH)).any():
            causes.append("wrong_class")
        elif (same & (row >= IOU_LOC)).any():
            causes.append("poor_box")
        else:
            causes.append("bg_present" if k in present else "bg_absent")
    return causes, idx


def size_bin(w: float, h: float) -> str:
    """tiny < 16^2 px, small < 32^2 (COCO small minus tiny), medium < 96^2, large otherwise (bbox area)."""
    a = w * h
    return "tiny" if a < 256 else "small" if a < 1024 else "medium" if a < 9216 else "large"


def density_bin(n_gt: int) -> str:
    """Objects per image, bins fixed a priori."""
    return "1-5" if n_gt <= 5 else "6-20" if n_gt <= 20 else "21-100" if n_gt <= 100 else ">100"
