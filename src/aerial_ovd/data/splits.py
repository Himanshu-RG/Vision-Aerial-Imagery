"""Where each evaluation split lives. LAE-80C dev/test (M2) and the external NWPU VHR-10 set (M11, D45).

All splits share the 80-class LAE-80C vocabulary and category ids. NWPU image ids are offset by NWPU_ID_OFFSET
so they never collide with LAE-80C ids.
"""

from __future__ import annotations

from pathlib import Path

NWPU_ID_OFFSET = 1_000_000
NWPU_SOURCE = "NWPU-VHR10"

# NWPU class index (readme order) -> LAE-80C category id (all DIOR-18C classes). 2 = ship: no single equivalent, dropped (D45).
NWPU_TO_LAE = {1: 1, 3: 10, 4: 5, 5: 11, 6: 7, 7: 3, 8: 4, 9: 8, 10: 15}
NWPU_NAMES = {1: "airplane", 2: "ship", 3: "storage tank", 4: "baseball diamond", 5: "tennis court",
              6: "basketball court", 7: "ground track field", 8: "harbor", 9: "bridge", 10: "vehicle"}

_SPLITS = {
    "dev": ("LAE-80C-clean/dev.json", "LAE-80C-images"),
    "test": ("LAE-80C-clean/test.json", "LAE-80C-images"),
    "nwpu": ("NWPU-VHR10/nwpu.json", "NWPU-VHR10/images"),
}
SPLITS = tuple(_SPLITS)


def gt_path(store: Path, split: str) -> Path:
    return Path(store) / "datasets" / _SPLITS[split][0]


def image_dir(store: Path, split: str) -> Path:
    return Path(store) / "datasets" / _SPLITS[split][1]


def source_space(source: str, cat_source: dict[int, str]) -> set[int]:
    """Category ids annotated for images of `source` (D15)."""
    if source == NWPU_SOURCE:
        return set(NWPU_TO_LAE.values())
    return {c for c, s in cat_source.items() if s == source}
