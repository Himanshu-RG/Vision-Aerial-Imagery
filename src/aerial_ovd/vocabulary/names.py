"""Class-name policy: how dataset category names become detector prompt text.

Policy "light" (decision D14, 2026-10-24):
  * lower-case everything;
  * split words that the source dataset glued together (DIOR style, e.g. 'groundtrackfield');
  * nothing else: no synonyms, no prefixes, slashes kept, overlapping classes kept as defined.
The SAME prompt names are used by every method. Evaluation always uses category ids.
"""

from __future__ import annotations

# Explicit table instead of a heuristic word splitter, so the policy is auditable.
CONCATENATED_SPLITS = {
    "groundtrackfield": "ground track field",
    "baseballfield": "baseball field",
    "basketballcourt": "basketball court",
    "storagetank": "storage tank",
    "tenniscourt": "tennis court",
    "trainstation": "train station",
    "golffield": "golf field",
}


def prompt_name(raw: str, policy: str = "light") -> str:
    if policy == "raw":
        return raw.strip()
    if policy != "light":
        raise ValueError(f"unknown naming policy {policy!r}")
    name = " ".join(raw.strip().lower().split())
    return CONCATENATED_SPLITS.get(name, name)


def build_prompt(names: list[str]) -> str:
    """Grounding DINO text input: 'name1. name2. ... nameN.'"""
    if not names:
        raise ValueError("cannot build a prompt from an empty vocabulary")
    return " ".join(f"{n}." for n in names)
