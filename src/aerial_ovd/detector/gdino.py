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


# ---------------------------------------------------------------------------------------------
# M9 diagnostic (D36): tiled inference. The image is split into overlapping tiles (same geometry as the
# M5 crops), each tile and (optionally) the full image is detected with the SAME vocabulary, boxes are
# mapped back to image coordinates and merged with class-wise NMS, keeping the top `num_select`.
# ---------------------------------------------------------------------------------------------

def merge_detections(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou: float, top_k: int):
    """Class-wise NMS, then the top_k highest scores. boxes: (N, 4) xyxy."""
    from torchvision.ops import batched_nms

    if len(scores) == 0:
        return boxes, scores, classes
    keep = batched_nms(torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(scores, dtype=torch.float32),
                       torch.as_tensor(classes, dtype=torch.int64), iou).numpy()
    keep = keep[np.argsort(-scores[keep], kind="stable")][:top_k]
    return boxes[keep], scores[keep], classes[keep]


def detect_tiled(det: "GroundingDINO", image, names: list[str], grid: int = 2, overlap: float = 0.25,
                 nms_iou: float = 0.5, include_full: bool = True) -> Detections:
    from aerial_ovd.vocabulary.selectors import crop_boxes

    W, H = image.size
    parts, timings, tokens = [], {"preprocess": 0.0, "forward": 0.0, "postprocess": 0.0}, 0
    views = ([(0, 0, W, H)] if include_full else []) + crop_boxes(W, H, grid, overlap)
    for (x0, y0, x1, y1) in views:
        d = det.detect(image if (x0, y0, x1, y1) == (0, 0, W, H) else image.crop((x0, y0, x1, y1)), names)
        b = d.boxes_xyxy.copy()
        b[:, [0, 2]] += x0
        b[:, [1, 3]] += y0
        parts.append((b, d.scores, d.class_index))
        tokens = d.n_text_tokens
        for k in timings:
            timings[k] += d.timings_ms[k]
    t0 = time.perf_counter()
    boxes = np.concatenate([p[0] for p in parts])
    scores = np.concatenate([p[1] for p in parts])
    classes = np.concatenate([p[2] for p in parts])
    boxes, scores, classes = merge_detections(boxes, scores, classes, nms_iou, det.num_select)
    timings["postprocess"] += (time.perf_counter() - t0) * 1e3
    timings["n_views"] = len(views)
    return Detections(boxes_xyxy=boxes.astype(np.float32), scores=scores.astype(np.float32),
                      class_index=classes.astype(np.int64), n_text_tokens=tokens, timings_ms=timings)
