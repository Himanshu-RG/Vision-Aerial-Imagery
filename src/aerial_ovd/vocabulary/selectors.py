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
