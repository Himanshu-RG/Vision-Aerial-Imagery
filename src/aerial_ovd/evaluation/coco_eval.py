"""COCO-style evaluation for LAE-80C with two protocols (decision D15).

* "standard":     plain COCO evaluation over all categories.
* "source_aware": per image, predictions whose category is outside the image's source label
                  space are removed before evaluation (those classes were never annotated
                  there). Source identity is used here, at evaluation time only.

AP is reported at several maxDets (D21): 100 (official-comparable) and 900 (dense; headline).
Categories without any ground truth in the evaluated images are excluded from the mean
(pycocotools convention, AP = -1).
"""

from __future__ import annotations

import contextlib
import io
from typing import Iterable, Mapping

import numpy as np

MAX_DETS = (100, 300, 900)


def _coco_from_dict(d: dict):
    from pycocotools.coco import COCO

    coco = COCO()
    coco.dataset = d
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def filter_source_aware(preds: Iterable[dict], image_space: Mapping[int, set[int]]) -> list[dict]:
    """Keep only predictions whose category is annotated for that image's source."""
    return [p for p in preds if p["category_id"] in image_space[p["image_id"]]]


def _ap_from_precision(prec: np.ndarray, t_idx=None, k_idx=None, a_idx=0, m_idx=-1) -> float:
    # prec shape: [T, R, K, A, M]
    p = prec[:, :, :, a_idx, m_idx]
    if t_idx is not None:
        p = p[[t_idx]]
    if k_idx is not None:
        p = p[:, :, [k_idx]]
    p = p[p > -1]
    return float(p.mean()) if p.size else float("nan")


def evaluate(gt: dict, preds: list[dict], image_ids: Iterable[int], protocol: str = "standard",
             image_space: Mapping[int, set[int]] | None = None, max_dets: tuple[int, ...] = MAX_DETS,
             per_class: bool = True) -> dict:
    """Evaluate `preds` (COCO result dicts: image_id, category_id, bbox xywh, score) on `image_ids`."""
    from pycocotools.cocoeval import COCOeval

    if protocol not in ("standard", "source_aware"):
        raise ValueError(f"unknown protocol {protocol!r}")
    ids = sorted(set(image_ids))
    idset = set(ids)
    sub = {
        "images": [im for im in gt["images"] if im["id"] in idset],
        "annotations": [a for a in gt["annotations"] if a["image_id"] in idset],
        "categories": gt["categories"],
    }
    if len(sub["images"]) != len(ids):
        raise ValueError("some image_ids are not in the ground truth")
    preds = [p for p in preds if p["image_id"] in idset]
    if protocol == "source_aware":
        if image_space is None:
            raise ValueError("source_aware protocol needs image_space")
        preds = filter_source_aware(preds, image_space)

    result = {"protocol": protocol, "n_images": len(ids), "n_predictions": len(preds),
              "n_gt": len(sub["annotations"]), "max_dets": list(max_dets)}
    names = {c["id"]: c["name"] for c in gt["categories"]}
    if not preds:
        result.update({"empty_predictions": True,
                       **{f"AP@{m}": 0.0 for m in max_dets}, **{f"AP50@{m}": 0.0 for m in max_dets},
                       **{f"AR@{m}": 0.0 for m in max_dets}})
        return result

    coco_gt = _coco_from_dict(sub)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes([dict(p) for p in preds])
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.imgIds = ids
        ev.params.maxDets = list(max_dets)
        ev.evaluate()
        ev.accumulate()
    prec, rec = ev.eval["precision"], ev.eval["recall"]  # [T,R,K,A,M], [T,K,A,M]
    t50 = int(np.argmin(np.abs(ev.params.iouThrs - 0.5)))
    t75 = int(np.argmin(np.abs(ev.params.iouThrs - 0.75)))
    for mi, m in enumerate(max_dets):
        result[f"AP@{m}"] = _ap_from_precision(prec, m_idx=mi)
        result[f"AP50@{m}"] = _ap_from_precision(prec, t_idx=t50, m_idx=mi)
        r = rec[:, :, 0, mi]
        result[f"AR@{m}"] = float(r[r > -1].mean()) if (r > -1).any() else float("nan")
    m_last = len(max_dets) - 1
    result["AP75"] = _ap_from_precision(prec, t_idx=t75, m_idx=m_last)
    for ai, area in enumerate(("small", "medium", "large"), start=1):
        result[f"AP50_{area}"] = _ap_from_precision(prec, t_idx=t50, a_idx=ai, m_idx=m_last)
        r = rec[t50, :, ai, m_last]
        result[f"R50_{area}"] = float(r[r > -1].mean()) if (r > -1).any() else float("nan")
    if per_class:
        pc = {}
        for ki, cid in enumerate(ev.params.catIds):
            ap50 = _ap_from_precision(prec, t_idx=t50, k_idx=ki, m_idx=m_last)
            if not np.isnan(ap50):
                pc[str(cid)] = {"name": names[cid], "AP50": ap50,
                                "AP": _ap_from_precision(prec, k_idx=ki, m_idx=m_last),
                                "R50": float(rec[t50, ki, 0, m_last])}
        result["per_class"] = pc
        result["n_classes_with_gt"] = len(pc)
    return result


# ---------------------------------------------------------------------------------------------
# Image-level bootstrap (M9). COCO matching is done per image, so the matches computed once by
# COCOeval can be resampled image-wise and re-accumulated cheaply. `ap50_from_matches` follows
# pycocotools' accumulate() exactly (101-point interpolation, mergesort by score) for one IoU
# threshold, area "all" and one maxDets value; a test checks it reproduces COCOeval's AP50.
# ---------------------------------------------------------------------------------------------

def per_image_matches(gt: dict, preds: list[dict], image_ids: Iterable[int], protocol: str = "standard",
                      image_space: Mapping[int, set[int]] | None = None, iou: float = 0.5,
                      max_det: int = 900) -> dict:
    """Per category, per image: (scores, is_tp, is_ignored_dt, n_gt_not_ignored) at one IoU."""
    from pycocotools.cocoeval import COCOeval

    ids = sorted(set(image_ids))
    idset = set(ids)
    sub = {"images": [im for im in gt["images"] if im["id"] in idset],
           "annotations": [a for a in gt["annotations"] if a["image_id"] in idset],
           "categories": gt["categories"]}
    preds = [p for p in preds if p["image_id"] in idset]
    if protocol == "source_aware":
        preds = filter_source_aware(preds, image_space)
    coco_gt = _coco_from_dict(sub)
    cat_ids = [c["id"] for c in gt["categories"]]
    empty = (np.zeros(0), np.zeros(0, bool), np.zeros(0, bool))
    out = {"image_ids": ids, "cat_ids": cat_ids, "per_cat": {}}
    if not preds:
        for k in cat_ids:
            n_gt = {i: len(coco_gt.getAnnIds(imgIds=[i], catIds=[k], iscrowd=None)) for i in ids}
            out["per_cat"][k] = {i: (*empty, n_gt[i]) for i in ids}
        return out
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes([dict(p) for p in preds])
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.imgIds = ids
        ev.params.maxDets = [max_det]
        ev.evaluate()
    t = int(np.argmin(np.abs(ev.params.iouThrs - iou)))
    n_img, n_area = len(ids), len(ev.params.areaRng)
    for ki, k in enumerate(ev.params.catIds):
        per = {}
        for ii, iid in enumerate(ids):
            e = ev.evalImgs[ki * n_area * n_img + 0 * n_img + ii]  # area index 0 = "all"
            if e is None:
                per[iid] = (*empty, 0)
                continue
            scores = np.asarray(e["dtScores"][:max_det], dtype=float)
            tp = np.asarray(e["dtMatches"][t, :max_det] > 0)
            ig = np.asarray(e["dtIgnore"][t, :max_det], dtype=bool)
            n_gt = int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))
            per[iid] = (scores, tp, ig, n_gt)
        out["per_cat"][k] = per
    return out


def _ap_one_category(scores, tp, ig, npig, rec_thrs=np.linspace(0.0, 1.00, 101)) -> float:
    order = np.argsort(-scores, kind="mergesort")
    tp, ig = tp[order], ig[order]
    tps = np.logical_and(tp, ~ig).cumsum()
    fps = np.logical_and(~tp, ~ig).cumsum()
    if len(tps) == 0:
        return 0.0
    rc = tps / npig
    pr = tps / (fps + tps + np.spacing(1))
    pr = np.maximum.accumulate(pr[::-1])[::-1]  # same as pycocotools' backward max loop
    q = np.zeros(len(rec_thrs))
    inds = np.searchsorted(rc, rec_thrs, side="left")
    valid = inds < len(pr)
    q[valid] = pr[inds[valid]]
    return float(q.mean())


def ap50_from_matches(m: dict, image_sample: list[int] | None = None) -> float:
    """Mean AP over categories with ground truth in `image_sample` (with repetitions allowed)."""
    sample = m["image_ids"] if image_sample is None else image_sample
    aps = []
    for k, per in m["per_cat"].items():
        parts = [per[i] for i in sample]
        npig = sum(p[3] for p in parts)
        if npig == 0:
            continue
        aps.append(_ap_one_category(np.concatenate([p[0] for p in parts]),
                                    np.concatenate([p[1] for p in parts]),
                                    np.concatenate([p[2] for p in parts]), npig))
    return float(np.mean(aps)) if aps else float("nan")


def paired_bootstrap(m_a: dict, m_b: dict, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Image-level paired bootstrap of AP50(b) - AP50(a). Both must cover the same images."""
    if m_a["image_ids"] != m_b["image_ids"]:
        raise ValueError("paired bootstrap needs the same image set for both conditions")
    ids = m_a["image_ids"]
    rng = np.random.default_rng(seed)
    point = ap50_from_matches(m_b) - ap50_from_matches(m_a)
    diffs = []
    for _ in range(n_boot):
        s = [ids[j] for j in rng.integers(0, len(ids), len(ids))]
        diffs.append(ap50_from_matches(m_b, s) - ap50_from_matches(m_a, s))
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    d = np.asarray(diffs)
    boot_mean, boot_sd = float(d.mean()), float(d.std(ddof=1))
    # Percentile CI is the pre-registered headline. Mean AP over categories is not a smooth statistic
    # (rare categories drop out of resamples), so the bootstrap distribution can be shifted; we also
    # report the bias and the "basic" (reverse-percentile) interval as a diagnostic.
    basic_lo, basic_hi = 2 * point - hi, 2 * point - lo
    return {"diff_point": point, "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot, "seed": seed,
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "boot_mean": boot_mean, "boot_sd": boot_sd, "bias": boot_mean - point,
            "bias_flag": bool(abs(boot_mean - point) > 0.25 * boot_sd) if boot_sd > 0 else False,
            "basic_ci_low": float(basic_lo), "basic_ci_high": float(basic_hi),
            "basic_ci_excludes_zero": bool(basic_lo > 0 or basic_hi < 0)}
