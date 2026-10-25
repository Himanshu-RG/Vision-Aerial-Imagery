"""Deployable vocabulary selectors (Module 4 baselines).

Every selector maps an image to one relevance score per master-vocabulary category; a budget k
then keeps the top-k. Rule D16: selectors see ONLY the decoded pixels -- the API takes a PIL image
and nothing else (no file name, source, original size or annotations).

Baselines:
  * FrequencySelector  -- fixed, image-independent prior (Baseline B "fixed reduced vocabulary"):
                          class image-frequency on the DEV split. In-sample on dev; fair on test.
  * RandomSelector     -- Baseline C; seeded per (seed, image content hash) so it is reproducible.
  * CLIPSelector       -- Baseline D "global CLIP": cosine similarity between the whole-image
                          embedding and prompt-ensembled class text embeddings. Scores are
                          similarities, NOT calibrated probabilities.
Text embeddings depend only on the vocabulary, so they are computed once and cached; image
embeddings are computed per image and included in the selector latency.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

# Fixed a priori (not tuned on any split): one remote-sensing, one aerial and one generic template.
DEFAULT_TEMPLATES = ("a satellite image of a {}.", "an aerial photo of a {}.", "a photo of a {}.")


class Selector:
    name = "base"

    def scores(self, image) -> np.ndarray:  # (n_classes,)
        raise NotImplementedError

    @staticmethod
    def top_k(scores: np.ndarray, k: int) -> np.ndarray:
        """Indices of the k highest scores; ties broken by lower index (stable, deterministic)."""
        if k < 1:
            raise ValueError("budget k must be >= 1")
        order = np.argsort(-scores, kind="stable")
        return np.sort(order[: min(k, len(scores))])


class FrequencySelector(Selector):
    name = "frequency"

    def __init__(self, class_image_counts: np.ndarray):
        self._s = np.asarray(class_image_counts, dtype=np.float64)

    def scores(self, image) -> np.ndarray:
        return self._s.copy()


class RandomSelector(Selector):
    def __init__(self, n_classes: int, seed: int):
        self.n, self.seed = n_classes, seed
        self.name = f"random_s{seed}"

    def scores(self, image) -> np.ndarray:
        digest = hashlib.sha256(image.tobytes()).digest()[:8]  # content-based, not file-name-based
        rng = np.random.default_rng([self.seed, int.from_bytes(digest, "little")])
        return rng.random(self.n)


class CLIPSelector(Selector):
    def __init__(self, name: str, arch: str, weights: str | Path, class_names: list[str],
                 templates=DEFAULT_TEMPLATES, device: str = "cpu", state_dict_format: bool = False):
        import open_clip
        import torch

        self.name, self.device = name, torch.device(device)
        if state_dict_format:  # e.g. RemoteCLIP: a plain state dict for the open_clip architecture
            model, _, preprocess = open_clip.create_model_and_transforms(arch)
            sd = torch.load(weights, map_location="cpu")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"{name}: state dict mismatch (missing={missing[:5]}, unexpected={unexpected[:5]})")
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=str(weights))
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.templates = tuple(templates)
        t0 = time.perf_counter()
        self.text_emb = self._encode_text(class_names)  # cached: depends only on the vocabulary
        self.text_cache_ms = (time.perf_counter() - t0) * 1e3

    def _encode_text(self, class_names):
        import torch

        with torch.no_grad():
            embs = []
            for name in class_names:
                tok = self.tokenizer([t.format(name) for t in self.templates]).to(self.device)
                e = self.model.encode_text(tok).float()
                e = e / e.norm(dim=-1, keepdim=True)
                e = e.mean(0)
                embs.append(e / e.norm())
            return torch.stack(embs)  # (C, D), unit norm

    def image_embedding(self, image):
        import torch

        with torch.no_grad():
            x = self.preprocess(image).unsqueeze(0).to(self.device)
            e = self.model.encode_image(x).float()
            return e / e.norm(dim=-1, keepdim=True)

    def scores(self, image) -> np.ndarray:
        e = self.image_embedding(image)
        return (e @ self.text_emb.T)[0].cpu().numpy().astype(np.float64)


def vocabulary_recall(selected: np.ndarray, gt: set[int]) -> float:
    """|selected ∩ GT| / |GT| for one image (GT non-empty: every LAE-80C image has >= 1 class)."""
    if not gt:
        raise ValueError("vocabulary recall is undefined for an image without GT classes")
    return len(set(selected.tolist()) & gt) / len(gt)


# ---------------------------------------------------------------------------------------------
# Module 5: global + local (crop) evidence.
# Every view (the whole image and each crop) is scored with CLIP; per-view class probabilities use
# the model's own logit scale (softmax over the master vocabulary), so global and local evidence are
# on the same scale. Views are stored once; combinations are evaluated offline (see m5 script).
# ---------------------------------------------------------------------------------------------

DEFAULT_GRIDS = (2, 3, 4)       # fixed a priori: 4 + 9 + 16 = 29 crops
DEFAULT_OVERLAP = 0.25          # each crop is 25% larger than its grid cell (overlap with neighbours)


def crop_boxes(width: int, height: int, grid: int, overlap: float = DEFAULT_OVERLAP) -> list[tuple[int, int, int, int]]:
    """Overlapping grid x grid crops covering the image; boxes (x0, y0, x1, y1) in pixels."""
    if grid < 1 or not 0 <= overlap < 1:
        raise ValueError("grid >= 1 and 0 <= overlap < 1 required")
    cw = min(width, round(width / grid * (1 + overlap)))
    ch = min(height, round(height / grid * (1 + overlap)))
    xs = [round(i * (width - cw) / (grid - 1)) for i in range(grid)] if grid > 1 else [0]
    ys = [round(j * (height - ch) / (grid - 1)) for j in range(grid)] if grid > 1 else [0]
    return [(x, y, x + cw, y + ch) for y in ys for x in xs]


def view_logits(sel: "CLIPSelector", image, grids=DEFAULT_GRIDS, overlap: float = DEFAULT_OVERLAP,
                batch: int = 32):
    """Scaled logits (logit_scale * cosine) for the whole image and every crop.

    Returns (global_logits (C,), crop_logits (V, C), crop_grid_ids (V,), boxes (V, 4)).
    Only the pixels are used (rule D16): crop geometry depends on the decoded image size only.
    """
    import torch

    W, H = image.size
    boxes, gids = [], []
    for g in grids:
        for b in crop_boxes(W, H, g, overlap):
            boxes.append(b)
            gids.append(g)
    views = [image] + [image.crop(b) for b in boxes]
    scale = float(sel.model.logit_scale.exp()) if hasattr(sel.model, "logit_scale") else 100.0
    out = []
    with torch.no_grad():
        for i in range(0, len(views), batch):
            x = torch.stack([sel.preprocess(v) for v in views[i:i + batch]]).to(sel.device)
            e = sel.model.encode_image(x).float()
            e = e / e.norm(dim=-1, keepdim=True)
            out.append((scale * e @ sel.text_emb.T).cpu())
    logits = torch.cat(out).numpy().astype(np.float32)
    return logits[0], logits[1:], np.array(gids, dtype=np.int16), np.array(boxes, dtype=np.int32)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def combine_views(global_logits: np.ndarray, crop_logits: np.ndarray, crop_grid_ids: np.ndarray,
                  alpha: float, agg: str = "max", grids=DEFAULT_GRIDS) -> np.ndarray:
    """score_c = alpha * p_global(c) + (1 - alpha) * agg_{crops in grids} p_crop(c).

    alpha = 1 -> global only; alpha = 0 -> local only. Probabilities are per-view softmax over classes.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    pg = softmax(global_logits.astype(np.float64))
    if alpha == 1.0:
        return pg
    mask = np.isin(crop_grid_ids, list(grids))
    if not mask.any():
        raise ValueError("no crops for the requested grids")
    pc = softmax(crop_logits[mask].astype(np.float64), axis=-1)
    if agg == "max":
        pl = pc.max(axis=0)
    elif agg == "mean":
        pl = pc.mean(axis=0)
    else:
        raise ValueError("agg must be 'max' or 'mean'")
    return alpha * pg + (1 - alpha) * pl
