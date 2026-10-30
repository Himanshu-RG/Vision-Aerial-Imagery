"""Module 6/7: variable-size vocabulary rules on top of per-class relevance probabilities.

All rules take, per image, the global class probabilities p_g (softmax over the master vocabulary of
RemoteCLIP logit_scale x cosine) and, where needed, the local (crop-mean) probabilities p_l. Each rule has a
single parameter that is calibrated on a CALIBRATION image set to reach a target mean vocabulary size
(decision D32) -- never to maximise a score -- so rules can be compared at matched size on held-out images.

Rules (each returns a boolean mask over classes, never empty):
  topk(k)              fixed budget (M4 baseline)
  threshold(t)         keep classes with p_g >= t                      (calibrated / conformal-style; adaptive)
  top_p(tau)           smallest prefix of classes by p_g with cumulative mass >= tau (adaptive)
  retain(k_base, d)    top-k_base by p_g, plus classes whose local evidence exceeds global by >= d
                       (global-local disagreement as uncertainty signal; D30)
All variable-size rules keep at least 1 and at most `max_k` classes.
"""

from __future__ import annotations

import numpy as np

MAX_K = 40


def _order(p: np.ndarray) -> np.ndarray:
    return np.argsort(-p, kind="stable")


def topk(p: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros(p.shape, bool)
    m[_order(p)[:max(1, min(k, len(p)))]] = True
    return m


def _clip_size(p: np.ndarray, m: np.ndarray, max_k: int) -> np.ndarray:
    n = int(m.sum())
    if n == 0:
        return topk(p, 1)
    if n > max_k:  # keep the max_k highest-probability members
        keep = [i for i in _order(p) if m[i]][:max_k]
        m = np.zeros(p.shape, bool)
        m[keep] = True
    return m


def threshold(p: np.ndarray, t: float, max_k: int = MAX_K) -> np.ndarray:
    return _clip_size(p, p >= t, max_k)


def top_p(p: np.ndarray, tau: float, max_k: int = MAX_K) -> np.ndarray:
    o = _order(p)
    c = np.cumsum(p[o])
    n = int(np.searchsorted(c, tau, side="left")) + 1
    m = np.zeros(p.shape, bool)
    m[o[:min(max(n, 1), len(p))]] = True
    return _clip_size(p, m, max_k)


def retain(p_global: np.ndarray, p_local: np.ndarray, k_base: int, d: float, max_k: int = MAX_K) -> np.ndarray:
    m = topk(p_global, k_base)
    m |= (p_local - p_global) >= d
    return _clip_size(p_global, m, max_k)


def calibrate(size_of_param, target: float, lo: float, hi: float, increasing: bool, iters: int = 60) -> float:
    """Bisection for a parameter whose mean-set-size function is monotone; returns the parameter whose
    mean size is closest to `target` (sizes are discrete, so exact equality is not always reachable)."""
    best = (abs(size_of_param(lo) - target), lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        s = size_of_param(mid)
        best = min(best, (abs(s - target), mid))
        if (s < target) == increasing:
            lo = mid
        else:
            hi = mid
    return best[1]
