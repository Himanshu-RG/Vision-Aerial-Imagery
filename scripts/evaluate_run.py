"""Evaluate a detection run under both protocols (D15) and write machine-readable metrics.

Usage:
    python scripts/evaluate_run.py outputs/predictions/<run_id>
Writes outputs/metrics/<run_id>/metrics.json (committed) and prints a summary table.
Failed or missing images stay in the denominator (they contribute no predictions).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aerial_ovd.evaluation.coco_eval import (  # noqa: E402
    ap50_from_matches, evaluate, paired_bootstrap, per_image_matches,
)
from aerial_ovd.utils.paths import get_store  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_detection import load_manifest, load_vocab  # noqa: E402


def read_records(path: Path) -> dict[int, dict]:
    """Last record per image wins (a resumed run may retry failed images)."""
    recs = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs[r["image_id"]] = r
    return recs


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--n-boot", type=int, default=1000, help="paired image-level bootstrap resamples (0 = off)")
    ap.add_argument("--reference", default="full", help="condition every other condition is compared with")
    args = ap.parse_args()
    meta = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))
    cfg, image_ids = meta["config"], meta["image_ids"]

    store = get_store()
    gt = json.loads((store / "datasets" / "LAE-80C-clean" / f"{cfg['split']}.json").read_text(encoding="utf-8"))
    rows = {int(r["image_id"]): r for r in load_manifest()}
    _, cat_source = load_vocab(cfg.get("naming_policy", "light"))
    image_space = {iid: {c for c, s in cat_source.items() if s == rows[iid]["source"]} for iid in image_ids}

    out = {"run_id": meta["run_id"], "host": meta["host"], "git_commit": meta["git_commit"],
           "device": cfg["detector"]["device"], "n_images": len(image_ids), "conditions": {}}
    for cond in cfg["conditions"]:
        path = args.run_dir / (cond.replace(":", "__") + ".jsonl")
        recs = read_records(path) if path.is_file() else {}
        ok = [r for iid, r in recs.items() if iid in set(image_ids) and r["status"] == "ok"]
        failed = [iid for iid in image_ids if recs.get(iid, {}).get("status") != "ok"]
        preds = [{"image_id": r["image_id"], "category_id": d[0], "score": d[1], "bbox": d[2:6]}
                 for r in ok for d in r["detections"]]
        lat = [r["latency_ms"] for r in ok]
        res = {
            "n_ok": len(ok), "n_failed_or_missing": len(failed), "failed_ids": failed[:50],
            "vocab_size_mean": statistics.fmean(len(r["vocab"]) for r in ok) if ok else None,
            "text_tokens_mean": statistics.fmean(r["n_text_tokens"] for r in ok) if ok else None,
            "latency_ms": {"median": pct(lat, 0.5), "p90": pct(lat, 0.9),
                           "forward_median": pct([r["timings_ms"]["forward"] for r in ok], 0.5)} if ok else None,
            "peak_mem_mb_max": max((r.get("peak_mem_mb", 0) for r in ok), default=None),
        }
        res["_matches"] = {}
        for proto in ("standard", "source_aware"):
            res[proto] = evaluate(gt, preds, image_ids, protocol=proto, image_space=image_space)
            m = per_image_matches(gt, preds, image_ids, protocol=proto, image_space=image_space)
            fast = ap50_from_matches(m)
            if abs(fast - res[proto]["AP50@900"]) > 1e-6:  # consistency check of the bootstrap machinery
                raise RuntimeError(f"fast AP50 {fast} != COCOeval {res[proto]['AP50@900']} ({cond}, {proto})")
            res["_matches"][proto] = m
        out["conditions"][cond] = res

    # Paired image-level bootstrap of AP50@900 differences vs the reference condition.
    ref = out["conditions"].get(args.reference)
    out["bootstrap"] = {"reference": args.reference, "metric": "AP50@900", "n_boot": args.n_boot}
    if ref is not None and args.n_boot > 0:
        for cond, r in out["conditions"].items():
            if cond == args.reference:
                continue
            out["bootstrap"][cond] = {
                proto: paired_bootstrap(ref["_matches"][proto], r["_matches"][proto], n_boot=args.n_boot, seed=0)
                for proto in ("standard", "source_aware")}
    for r in out["conditions"].values():
        r.pop("_matches")

    dst = REPO_ROOT / "outputs" / "metrics" / meta["run_id"]
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "metrics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    # Provenance: the resolved config, env snapshot, image sample and checkpoint lock travel with the metrics.
    (dst / "run_config.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")

    hdr = f"{'condition':<15}{'vocab':>6}{'fail':>5}{'lat ms':>8} | " + " | ".join(
        f"{p[:8]:>8} AP50@900 AP@900 AP50@100" for p in ("standard", "source_aware"))
    print(hdr)
    for cond, r in out["conditions"].items():
        cells = [f"{cond:<15}{(r['vocab_size_mean'] or 0):>6.1f}{r['n_failed_or_missing']:>5}"
                 f"{(r['latency_ms'] or {}).get('median') or 0:>8.0f}"]
        for p in ("standard", "source_aware"):
            m = r[p]
            cells.append(f"{'':>8} {m['AP50@900']:>8.3f} {m['AP@900']:>6.3f} {m['AP50@100']:>8.3f}")
        print(" | ".join(cells))
    for cond, b in out["bootstrap"].items():
        if isinstance(b, dict):
            print(f"{cond:<15} minus {args.reference}: " + "  ".join(
                f"{p}: {v['diff_point']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]" for p, v in b.items()))
    print(f"\nwritten: {dst / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
