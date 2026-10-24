"""LAE-80C loading, cleaning and manifest building.

LAE-80C is a COCO-format union of five sources whose class sets are DISJOINT
(DIOR-18C, DOTA-6C, FAIR-18C, PP-4C, Xview-34C). Each image is annotated only for its
own source's classes, so the source label space matters for evaluation (see
`source_label_space`). Source identity is derived from the file-name prefix and must only
ever be used by evaluation code, never by vocabulary selectors.
"""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KNOWN_SOURCES = ("DIOR-18C", "DOTA-6C", "FAIR-18C", "PP-4C", "Xview-34C")


def load_coco(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        coco = json.load(f)
    for key in ("images", "annotations", "categories"):
        if key not in coco:
            raise ValueError(f"{path}: missing COCO key {key!r}")
    return coco


def source_of(file_name: str) -> str:
    """'FAIR-18C_00012.jpg' -> 'FAIR-18C'. Raises on unknown prefixes instead of guessing."""
    prefix = file_name.split("_", 1)[0]
    if prefix not in KNOWN_SOURCES:
        raise ValueError(f"unknown source prefix {prefix!r} in {file_name!r}")
    return prefix


def source_label_space(coco: dict[str, Any]) -> dict[str, set[int]]:
    """Category ids annotated per source (computed from the annotations themselves)."""
    src = {im["id"]: source_of(im["file_name"]) for im in coco["images"]}
    space: dict[str, set[int]] = defaultdict(set)
    for a in coco["annotations"]:
        space[src[a["image_id"]]].add(a["category_id"])
    return dict(space)


@dataclass
class CleaningReport:
    n_images: int = 0
    n_annotations_in: int = 0
    n_annotations_out: int = 0
    clipped_min_area_kept: float = 1.0
    counts: Counter = field(default_factory=Counter)
    examples: dict[str, list] = field(default_factory=lambda: defaultdict(list))

    def note(self, kind: str, example: Any, keep: int = 5) -> None:
        self.counts[kind] += 1
        if len(self.examples[kind]) < keep:
            self.examples[kind].append(example)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_images": self.n_images,
            "n_annotations_in": self.n_annotations_in,
            "n_annotations_out": self.n_annotations_out,
            "clipped_min_area_kept": round(self.clipped_min_area_kept, 3),
            "counts": dict(sorted(self.counts.items())),
            "examples": {k: v for k, v in sorted(self.examples.items())},
        }


def clean_coco(
    coco: dict[str, Any],
    clip_to_image: bool = True,
    drop_exact_duplicates: bool = True,
    drop_nonpositive: bool = True,
    tiny_side_flag_px: float = 2.0,
) -> tuple[dict[str, Any], CleaningReport]:
    """Return a cleaned deep copy of `coco` plus a report of every change and flag.

    Nothing is dropped silently: every modification is counted in the report.
    """
    out = copy.deepcopy(coco)
    rep = CleaningReport(n_images=len(out["images"]), n_annotations_in=len(out["annotations"]))

    images = {im["id"]: im for im in out["images"]}
    cat_ids = {c["id"] for c in out["categories"]}
    if len(images) != len(out["images"]):
        rep.note("duplicate_image_ids", len(out["images"]) - len(images))
    names = Counter(im["file_name"] for im in out["images"])
    for name, c in names.items():
        if c > 1:
            rep.note("duplicate_file_names", name)
    ann_ids = Counter(a["id"] for a in out["annotations"])
    for aid, c in ann_ids.items():
        if c > 1:
            rep.note("duplicate_annotation_ids", aid)

    kept, seen = [], set()
    for a in out["annotations"]:
        im = images.get(a["image_id"])
        if im is None:
            rep.note("dropped_unknown_image", a["id"])
            continue
        if a["category_id"] not in cat_ids:
            rep.note("dropped_unknown_category", a["id"])
            continue
        if "file_name" in a and a["file_name"] != im["file_name"]:
            rep.note("annotation_file_name_mismatch", a["id"])
        if a.get("iscrowd", 0):
            rep.note("iscrowd", a["id"])

        x, y, w, h = (float(v) for v in a["bbox"])
        W, H = float(im["width"]), float(im["height"])
        if clip_to_image and (x < 0 or y < 0 or x + w > W or y + h > H):
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(W, x + w), min(H, y + h)
            kept_frac = max(0.0, x1 - x0) * max(0.0, y1 - y0) / (w * h) if w > 0 and h > 0 else 0.0
            rep.note("clipped_to_image", {"id": a["id"], "area_kept": round(kept_frac, 3)})
            rep.clipped_min_area_kept = min(rep.clipped_min_area_kept, kept_frac)
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            rep.note("nonpositive_box", a["id"])
            if drop_nonpositive:
                continue
        key = (a["image_id"], a["category_id"], round(x, 1), round(y, 1), round(w, 1), round(h, 1))
        if key in seen:
            rep.note("exact_duplicate", a["id"])
            if drop_exact_duplicates:
                continue
        seen.add(key)
        if min(w, h) < tiny_side_flag_px:
            rep.note("tiny_box_flagged", a["id"])

        a["bbox"] = [x, y, w, h]
        a["area"] = w * h  # horizontal-box area; original polygon areas are not used for COCO bins
        a.pop("segmentation", None)  # xView polygons are axis-aligned rectangles; boxes are the GT
        a["iscrowd"] = int(a.get("iscrowd", 0))
        kept.append(a)

    out["annotations"] = kept
    rep.n_annotations_out = len(kept)
    annotated = {a["image_id"] for a in kept}
    for iid in images:
        if iid not in annotated:
            rep.note("image_without_annotations", iid)
    return out, rep


def build_manifest(coco: dict[str, Any], size_bins: tuple[float, float] = (1024.0, 9216.0)) -> list[dict[str, Any]]:
    """One row per image: identity, source, size, class presence and object-size counts."""
    small, large = size_bins
    per_img: dict[int, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        per_img[a["image_id"]].append(a)
    rows = []
    for im in sorted(coco["images"], key=lambda r: r["id"]):
        anns = per_img.get(im["id"], [])
        areas = [a["bbox"][2] * a["bbox"][3] for a in anns]
        classes = sorted({a["category_id"] for a in anns})
        rows.append({
            "image_id": im["id"],
            "file_name": im["file_name"],
            "source": source_of(im["file_name"]),
            "width": im["width"],
            "height": im["height"],
            "n_boxes": len(anns),
            "n_classes": len(classes),
            "classes": " ".join(map(str, classes)),
            "n_small": sum(ar < small for ar in areas),
            "n_medium": sum(small <= ar < large for ar in areas),
            "n_large": sum(ar >= large for ar in areas),
            "n_tiny_flag": sum(min(a["bbox"][2], a["bbox"][3]) < 2 for a in anns),
        })
    return rows


def subset_coco(coco: dict[str, Any], image_ids: set[int]) -> dict[str, Any]:
    """COCO dict restricted to `image_ids` (categories kept complete)."""
    return {
        **{k: v for k, v in coco.items() if k not in ("images", "annotations")},
        "images": [im for im in coco["images"] if im["id"] in image_ids],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in image_ids],
    }
