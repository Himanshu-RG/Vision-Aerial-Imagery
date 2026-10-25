"""Vocabulary-selection quality, independent of the detector (Module 4, step 1).

Usage:
    python scripts/evaluate_selection.py --split dev
Reads outputs/selection/<split>/*.npz. Writes
    outputs/metrics/selection_<split>/selection_metrics.json   (committed)
    reports/generated/M4_selection_tables.md                    (generated tables)
    outputs/figures/m4_vocab_recall_<split>.png                 (recall vs budget)

Conventions (M9):
  * vocabulary recall (image) = |selected ∩ GT classes| / |GT classes|; reported as the mean over
    images (every LAE-80C image has >= 1 GT class, so no image is excluded);
  * full coverage = share of images whose GT classes are ALL selected;
  * pair recall = share of (image, GT class) pairs selected, broken down by class frequency band
    (dev image-count terciles) and by whether that class's objects in the image are mostly small
    (median box area < 32^2 px);
  * irrelevant rejection = share of the image's non-GT classes NOT selected;
  * ranking mAP = multi-label (class-wise) average precision of the scores, budget-free.
Selection recall is NOT detection recall.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aerial_ovd.utils.paths import get_store  # noqa: E402
from run_detection import load_manifest  # noqa: E402

BUDGETS = [1, 2, 3, 5, 8, 12, 20, 40, 80]
MAN = REPO_ROOT / "data" / "manifests"


def class_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision of ranking images by score for one class (labels in {0,1})."""
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = labels[order]
    prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((prec * hits).sum() / hits.sum())


def paired_mean_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d = b - a
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "excludes_zero": bool(lo > 0 or hi < 0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

    sel_dir = REPO_ROOT / "outputs" / "selection" / args.split
    runs = {p.stem: np.load(p, allow_pickle=False) for p in sorted(sel_dir.glob("*.npz"))}
    if not runs:
        print(f"no selection outputs in {sel_dir}", file=sys.stderr)
        return 1
    rows = {int(r["image_id"]): r for r in load_manifest() if r["split"] == args.split}
    vocab = list(csv.DictReader((MAN / "lae80c_vocabulary.csv").open(encoding="utf-8")))
    cat_ids = np.array([int(v["category_id"]) for v in vocab])
    cidx = {c: i for i, c in enumerate(cat_ids)}

    # frequency bands (dev image counts, terciles) and small-dominated (image, class) pairs
    dev_cnt = np.array([int(v["n_images_dev"]) for v in vocab])
    t1, t2 = np.quantile(dev_cnt[dev_cnt > 0], [1 / 3, 2 / 3])
    band = {int(v["category_id"]): ("rare" if c <= t1 else "common" if c > t2 else "mid")
            for v, c in zip(vocab, dev_cnt)}
    gt = json.loads((get_store() / "datasets" / "LAE-80C-clean" / f"{args.split}.json").read_text(encoding="utf-8"))
    areas = defaultdict(list)
    for a in gt["annotations"]:
        areas[(a["image_id"], a["category_id"])].append(a["bbox"][2] * a["bbox"][3])
    small_pair = {k: float(np.median(v)) < 1024 for k, v in areas.items()}

    out = {"split": args.split, "budgets": BUDGETS, "freq_band_thresholds_dev_images": [float(t1), float(t2)],
           "selectors": {}}
    per_image_recall = {}
    for name, z in runs.items():
        meta = json.loads(str(z["meta"]))
        ids, S = z["image_ids"], z["scores"]
        assert list(z["cat_ids"]) == list(cat_ids), "category order mismatch"
        valid = ~np.isnan(S).any(axis=1)
        G = np.zeros_like(S, dtype=bool)
        for i, iid in enumerate(ids):
            for c in rows[int(iid)]["classes"].split():
                G[i, cidx[int(c)]] = True
        res = {"n_images": int(len(ids)), "n_failed": int((~valid).sum()), "meta": meta, "by_budget": {}}
        ranks = np.argsort(np.argsort(-S, axis=1, kind="stable"), axis=1)  # 0 = best
        for k in BUDGETS:
            sel = ranks < k  # failed images (NaN) select nothing -> recall 0, kept in denominator
            sel[~valid] = False
            hits = (sel & G).sum(1)
            ng = G.sum(1)
            rec = hits / ng
            pairs = np.argwhere(G)
            pr = defaultdict(list)
            for i, j in pairs:
                key_small = "small" if small_pair.get((int(ids[i]), int(cat_ids[j])), False) else "not_small"
                pr[band[int(cat_ids[j])]].append(sel[i, j])
                pr[key_small].append(sel[i, j])
            res["by_budget"][str(k)] = {
                "vocab_recall_mean": float(rec.mean()),
                "full_coverage": float((hits == ng).mean()),
                "pair_recall": float(sel[G].mean()),
                "irrelevant_rejection": float((((~sel) & ~G).sum(1) / (~G).sum(1)).mean()),
                **{f"pair_recall_{b}": float(np.mean(v)) for b, v in sorted(pr.items())},
            }
            per_image_recall[(name, k)] = rec
        aps = [class_ap(S[valid, j], G[valid, j]) for j in range(len(cat_ids))]
        res["ranking_mAP"] = float(np.nanmean(aps))
        res["gt_class_median_rank"] = float(np.median(ranks[G]) + 1)
        res["selector_ms_median"] = float(np.median(z["image_ms"][valid])) if valid.any() else None
        out["selectors"][name] = res

    # random: mean +- sd over seeds
    rnd = [n for n in runs if n.startswith("random_s")]
    if rnd:
        out["random_summary"] = {str(k): {"vocab_recall_mean": float(np.mean([out["selectors"][n]["by_budget"][str(k)]["vocab_recall_mean"] for n in rnd])),
                                          "sd": float(np.std([out["selectors"][n]["by_budget"][str(k)]["vocab_recall_mean"] for n in rnd]))}
                                 for k in BUDGETS}
    # paired bootstrap of recall@k: RemoteCLIP vs OpenAI CLIP, and each CLIP vs frequency prior
    comps = [("clip_openai_b32", "remoteclip_b32"), ("frequency", "clip_openai_b32"), ("frequency", "remoteclip_b32")]
    out["bootstrap_recall"] = {}
    for a, b in comps:
        if a in runs and b in runs:
            out["bootstrap_recall"][f"{b}_minus_{a}"] = {
                str(k): paired_mean_bootstrap(per_image_recall[(a, k)], per_image_recall[(b, k)]) for k in (1, 2, 3, 5, 8)}

    mdir = REPO_ROOT / "outputs" / "metrics" / f"selection_{args.split}"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "selection_metrics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")

    # ---- generated tables ----
    names = [n for n in ["frequency", "clip_openai_b32", "remoteclip_b32"] if n in runs]
    L = [f"# M4 selection tables ({args.split}; generated by `scripts/evaluate_selection.py`)", "",
         "Vocabulary recall = mean over images of |selected ∩ GT| / |GT|. Random = mean ± sd over 3 seeds. "
         "Selector time is CPU on the dev laptop (not a latency result).", "",
         "## Vocabulary recall by budget k", "",
         "| selector | " + " | ".join(f"k={k}" for k in BUDGETS) + " | ranking mAP | median GT rank | ms/img (CPU) |",
         "|---|" + "---|" * (len(BUDGETS) + 3)]
    for n in names:
        r = out["selectors"][n]
        L.append(f"| {n} | " + " | ".join(f"{r['by_budget'][str(k)]['vocab_recall_mean']:.3f}" for k in BUDGETS)
                 + f" | {r['ranking_mAP']:.3f} | {r['gt_class_median_rank']:.0f} | {r['selector_ms_median']:.1f} |")
    if rnd:
        L.append("| random (3 seeds) | " + " | ".join(
            f"{out['random_summary'][str(k)]['vocab_recall_mean']:.3f}±{out['random_summary'][str(k)]['sd']:.3f}" for k in BUDGETS)
            + " | – | – | – |")
    L += ["", "## Full coverage (all GT classes selected) and irrelevant-class rejection", "",
          "| selector | " + " | ".join(f"cov k={k}" for k in (2, 3, 5, 8, 12)) + " | " + " | ".join(f"rej k={k}" for k in (3, 8)) + " |",
          "|---|" + "---|" * 7]
    for n in names:
        b = out["selectors"][n]["by_budget"]
        L.append(f"| {n} | " + " | ".join(f"{b[str(k)]['full_coverage']:.3f}" for k in (2, 3, 5, 8, 12)) + " | "
                 + " | ".join(f"{b[str(k)]['irrelevant_rejection']:.3f}" for k in (3, 8)) + " |")
    L += ["", f"## Pair recall at k=5 by class frequency band (dev image-count terciles {t1:.0f}/{t2:.0f}) and object size", "",
          "| selector | rare | mid | common | small-dominated | not small |", "|---|---|---|---|---|---|"]
    for n in names:
        b = out["selectors"][n]["by_budget"]["5"]
        L.append(f"| {n} | " + " | ".join(f"{b.get('pair_recall_' + x, float('nan')):.3f}"
                                          for x in ("rare", "mid", "common", "small", "not_small")) + " |")
    L += ["", "## Paired bootstrap of vocabulary recall (2000 resamples, 95% CI)", "",
          "| comparison | " + " | ".join(f"k={k}" for k in (1, 2, 3, 5, 8)) + " |", "|---|" + "---|" * 5]
    for comp, d in out["bootstrap_recall"].items():
        L.append(f"| {comp} | " + " | ".join(f"{v['diff']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]"
                                             + ("*" if v["excludes_zero"] else "") for v in d.values()) + " |")
    gen = REPO_ROOT / "reports" / "generated" / "M4_selection_tables.md"
    gen.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")

    # ---- figure ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for n in names:
        ax.plot(BUDGETS, [out["selectors"][n]["by_budget"][str(k)]["vocab_recall_mean"] for k in BUDGETS], marker="o", label=n)
    if rnd:
        ax.plot(BUDGETS, [out["random_summary"][str(k)]["vocab_recall_mean"] for k in BUDGETS], ls="--", color="gray", label="random (3 seeds)")
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS, [str(k) for k in BUDGETS])
    ax.set_xlabel("vocabulary budget k (classes kept, of 80)")
    ax.set_ylabel("mean vocabulary recall")
    ax.set_title(f"LAE-80C {args.split}: does the selector keep the classes present?")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fdir = REPO_ROOT / "outputs" / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fdir / f"m4_vocab_recall_{args.split}.png", dpi=150)
    print(gen.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
