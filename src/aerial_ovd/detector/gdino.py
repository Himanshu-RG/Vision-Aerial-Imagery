"""Frozen Grounding DINO (HF transformers) with official-style class scoring.

Class scores follow the official COCO evaluation of Grounding DINO
(IDEA-Research/GroundingDINO, demo/test_ap_on_coco.py, PostProcessCocoGrounding):
  * caption  = " . ".join(names) + " ."   (lower-case names)
  * positive map: for each class, the tokens of its name; each row normalised to sum 1
  * class prob = sigmoid(token logits) @ positive_map.T   -> mean over the name's tokens
  * keep the top-`num_select` (query, class) pairs per image, boxes cxcywh -> xyxy in pixels.
`class_reduce="max"` (max over the name's tokens) is available only as an ablation (D20).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

MAX_TEXT_LEN = 256


def build_caption(names: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Official caption format plus the character span of every name."""
    if not names:
        raise ValueError("empty vocabulary")
    caption, spans = "", []
    for i, name in enumerate(names):
        name = name.lower()
        start = len(caption)
        caption += name
        spans.append((start, len(caption)))
        caption += " . " if i < len(names) - 1 else " ."
    return caption, spans


def positive_map_from_spans(encoding, spans: list[tuple[int, int]], max_len: int = MAX_TEXT_LEN) -> torch.Tensor:
    """(n_classes, max_len) map: 1 on the tokens of each name, rows normalised to sum 1.

    Mirrors groundingdino.util.vl_utils.create_positive_map_from_span, including its
    fallback of trying the next characters when a span starts or ends on whitespace.
    """
    pmap = torch.zeros((len(spans), max_len), dtype=torch.float32)
    for j, (beg, end) in enumerate(spans):
        beg_pos = next((p for p in (encoding.char_to_token(beg + d) for d in range(3)) if p is not None), None)
        end_pos = next((p for p in (encoding.char_to_token(end - 1 - d) for d in range(3)) if p is not None), None)
        if beg_pos is None or end_pos is None:
            raise ValueError(f"could not map characters {beg}:{end} to tokens")
        if end_pos >= max_len:
            raise ValueError(f"class span ends at token {end_pos} >= max_text_len {max_len} (prompt too long)")
        pmap[j, beg_pos:end_pos + 1] = 1.0
    return pmap / (pmap.sum(-1, keepdim=True) + 1e-6)


@dataclass
class Detections:
    boxes_xyxy: np.ndarray        # (N, 4) float32, pixels of the original image
    scores: np.ndarray            # (N,)
    class_index: np.ndarray       # (N,) index into the vocabulary passed in
    n_text_tokens: int = 0
    timings_ms: dict = field(default_factory=dict)


class GroundingDINO:
    def __init__(self, checkpoint_dir: str | Path, device: str = "cpu", precision: str = "fp32",
                 num_select: int = 900, class_reduce: str = "mean"):
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        if class_reduce not in ("mean", "max"):
            raise ValueError("class_reduce must be 'mean' (official) or 'max' (ablation)")
        if precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"unknown precision {precision}")
        self.device = torch.device(device)
        self.precision = precision
        self.num_select = num_select
        self.class_reduce = class_reduce
        self.processor = AutoProcessor.from_pretrained(checkpoint_dir)
        self.model = GroundingDinoForObjectDetection.from_pretrained(checkpoint_dir).to(self.device).eval()
        self.max_text_len = self.model.config.max_text_len

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def detect(self, image, names: list[str]) -> Detections:
        """image: PIL RGB image. names: prompt names of the vocabulary (order = class_index)."""
        t0 = time.perf_counter()
        caption, spans = build_caption(names)
        enc = self.processor.tokenizer(caption)
        n_tokens = len(enc["input_ids"])
        if n_tokens > self.max_text_len:
            raise ValueError(f"prompt has {n_tokens} tokens > {self.max_text_len}; chunking is not enabled (D10)")
        pmap = positive_map_from_spans(enc, spans, self.max_text_len).to(self.device)
        inputs = self.processor(images=image, text=caption, return_tensors="pt").to(self.device)
        self._sync()
        t1 = time.perf_counter()

        autocast = (torch.autocast(device_type=self.device.type,
                                   dtype=torch.float16 if self.precision == "fp16" else torch.bfloat16)
                    if self.precision != "fp32" else torch.autocast(device_type=self.device.type, enabled=False))
        with autocast:
            out = self.model(**inputs)
        self._sync()
        t2 = time.perf_counter()

        logits = out.logits[0].float()                     # (Q, max_text_len); -inf beyond the text
        prob_tok = logits.sigmoid()
        if self.class_reduce == "mean":
            prob = prob_tok @ pmap.T                        # (Q, C)
        else:
            mask = pmap > 0
            prob = torch.stack([prob_tok[:, m].max(dim=1).values for m in mask], dim=1)
        k = min(self.num_select, prob.numel())
        scores, idx = torch.topk(prob.flatten(), k)
        q_idx, c_idx = idx // prob.shape[1], idx % prob.shape[1]
        cx, cy, w, h = out.pred_boxes[0].float()[q_idx].unbind(-1)
        W, H = image.size
        boxes = torch.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], dim=-1)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H)
        self._sync()
        t3 = time.perf_counter()
        return Detections(
            boxes_xyxy=boxes.cpu().numpy().astype(np.float32),
            scores=scores.cpu().numpy().astype(np.float32),
            class_index=c_idx.cpu().numpy().astype(np.int64),
            n_text_tokens=n_tokens,
            timings_ms={"preprocess": (t1 - t0) * 1e3, "forward": (t2 - t1) * 1e3, "postprocess": (t3 - t2) * 1e3},
        )
