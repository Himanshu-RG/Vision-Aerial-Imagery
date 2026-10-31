"""M10: failure analysis of a detection run (dev only). Needs the run's predictions (GPU machine) and the dev GT.

Usage:
    python scripts/m10_failure_analysis.py outputs/predictions/<run_id> [--conditions ...] [--thr 0.14] [--no-figures]
Writes (all small, committed):
    outputs/metrics/m10_failure/<run_id>/failure.json
    reports/generated/M10_failure_tables.md
    reports/figures/m10/*.jpg          (qualitative examples, chosen by fixed rules below, not by hand)
Definitions of outcomes and causes: src/aerial_ovd/evaluation/failure.py. Score threshold 0.14 = D35's standard-
protocol operating point; "any" = all top-900 detections (threshold-free ceiling). Standard protocol (all classes).
"FP of out-of-source class" = share of detections (not TP) whose class is never annotated in the image's source
dataset (D15 label-space artefact; such detections may be correct but unlabelled). Strata fixed a priori: object size (tiny < 16^2 px), objects per image, class rarity (terciles of the number of
dev images containing the class, from the committed vocabulary manifest), source dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aerial_ovd.evaluation.coco_eval import _ap_one_category, ap50_from_matches, per_image_matches  # noqa: E402
from aerial_ovd.evaluation.failure import (  # noqa: E402
    DET_CAUSES, GT_OUTCOMES, density_bin, det_causes, gt_outcomes, size_bin,
)
from aerial_ovd.utils.paths import get_store  # noqa: E402
from evaluate_run import read_records  # noqa: E402
from run_detection import load_manifest  # noqa: E402

DEFAULT_CONDS = ["full", "oracle_gt", "sel:remoteclip_b32:k8", "set:m6_threshold_m8", "set:e5_caption_thr_m8"]
HEADLINE = "set:m6_threshold_m8"
SIZE_ORDER, DENS_ORDER, RARE_ORDER = ["tiny", "small", "medium", "large"], ["1-5", "6-20", "21-100", ">100"], ["rare", "common", "frequent"]


def rarity_bins(vocab_rows):
    n = np.array([int(v["n_images_dev"]) for v in vocab_rows])
    q1, q2 = np.quantile(n, [1 / 3, 2 / 3])
    return ({int(v["category_id"]): ("rare" if int(v["n_images_dev"]) <= q1 else "common" if int(v["n_images_dev"]) <= q2 else "frequent")
             for v in vocab_rows}, [float(q1), float(q2)])


def share(counter: Counter, keys) -> dict:
    tot = sum(v for k, v in counter.items() if not k.startswith("_"))
    out = {"n": tot, **{k: round(counter[k] / tot, 4) if tot else None for k in keys}}
    if "_out_of_source" in counter:
        out["fp_out_of_source"] = round(counter["_out_of_source"] / tot, 4) if tot else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDS)
    ap.add_argument("--thr", type=float, default=0.14)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    meta = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))
    split, image_ids = meta["config"]["split"], meta["image_ids"]
    if split != "dev":
        raise SystemExit("M10 is a dev-only analysis (no test leakage)")
    store = get_store()
    gt = json.loads((store / "datasets" / "LAE-80C-clean" / f"{split}.json").read_text(encoding="utf-8"))
    rows = {int(r["image_id"]): r for r in load_manifest()}
    vocab_rows = list(csv.DictReader((REPO_ROOT / "data/manifests/lae80c_vocabulary.csv").open(encoding="utf-8")))
    name = {int(v["category_id"]): v["raw_name"] for v in vocab_rows}
    cat_src = {int(v["category_id"]): v["source"] for v in vocab_rows}
    rarity, rarity_cuts = rarity_bins(vocab_rows)
    ids = set(image_ids)
    anns = defaultdict(list)
    for a in gt["annotations"]:
        if a["image_id"] in ids:
            anns[a["image_id"]].append(a)
    G = {i: (np.array([a["bbox"] for a in anns[i]], float).reshape(-1, 4), np.array([a["category_id"] for a in anns[i]], int))
         for i in image_ids}
    dens = {i: density_bin(len(G[i][1])) if len(G[i][1]) else "0" for i in image_ids}

    res, per_image = {}, {}
    for cond in args.conditions:
        path = args.run_dir / (cond.replace(":", "__") + ".jsonl")
        recs = read_records(path)
        out = {"gt": {}, "det_causes": {}, "confusions": {}, "ap50_by_density": {}, "ap50_by_rarity": {}}
        img_stats = {}
        for tag, thr in (("any", 0.0), (f"@{args.thr:g}", args.thr)):
            strata = defaultdict(Counter)
            conf = Counter()
            for i in image_ids:
                r = recs.get(i)
                gb, gc = G[i]
                if not len(gc):
                    continue
                if r is None or r.get("status") != "ok":
                    outs = [("missed", None)] * len(gc)
                    vocab = set()
                else:
                    d = np.array(r["detections"], float).reshape(-1, 6)
                    vocab = set(r["vocab"])
                    outs = gt_outcomes(gb, gc, d[:, 2:6], d[:, 0].astype(int), d[:, 1], vocab, thr)
                src = rows[i]["source"]
                for (o, cf), b, c in zip(outs, gb, gc):
                    for key in ("all", "size:" + size_bin(b[2], b[3]), "density:" + dens[i], "rarity:" + rarity[int(c)], "source:" + src):
                        strata[key][o] += 1
                    if cf is not None:
                        conf[(int(c), cf, o)] += 1
                if tag != "any":
                    img_stats[i] = Counter(o for o, _ in outs)
                    img_stats[i]["tiny_missed"] = sum(1 for (o, _), b in zip(outs, gb) if o != "detected" and size_bin(b[2], b[3]) == "tiny")
                    img_stats[i]["rare_vocab_miss"] = sum(1 for (o, _), c in zip(outs, gc) if o == "vocab_miss" and rarity[int(c)] == "rare")
            out["gt"][tag] = {k: share(v, GT_OUTCOMES) for k, v in sorted(strata.items())}
            out["confusions"][tag] = [{"gt": name[g], "predicted_as": name[p], "outcome": o, "n": n}
                                      for (g, p, o), n in conf.most_common(15)]
        causes = Counter()
        causes_src = defaultdict(Counter)
        for i in image_ids:
            r = recs.get(i)
            if r is None or r.get("status") != "ok":
                continue
            d = np.array(r["detections"], float).reshape(-1, 6)
            cs, idx = det_causes(*G[i], d[:, 2:6], d[:, 0].astype(int), d[:, 1], args.thr)
            causes.update(cs)
            causes_src[rows[i]["source"]].update(cs)
            # detections of a class that is never annotated in this image's source (label-space artefact, D15)
            oos = sum(1 for c, j in zip(cs, idx) if c != "tp" and cat_src[int(d[j, 0])] != rows[i]["source"])
            causes["_out_of_source"] += oos
            causes_src[rows[i]["source"]]["_out_of_source"] += oos
            img_stats.setdefault(i, Counter())["tp_thr"] = cs.count("tp")
        out["det_causes"][f"@{args.thr:g}"] = {"all": share(causes, DET_CAUSES),
                                               **{s: share(c, DET_CAUSES) for s, c in sorted(causes_src.items())}}
        # AP50 (standard, maxDets 900) on image subsets by density; per-class AP50 grouped by rarity
        preds = [{"image_id": r["image_id"], "category_id": d[0], "score": d[1], "bbox": d[2:6]}
                 for r in recs.values() if r.get("status") == "ok" and r["image_id"] in ids for d in r["detections"]]
        m = per_image_matches(gt, preds, image_ids)
        for b in DENS_ORDER:
            sub = [i for i in image_ids if dens[i] == b]
            out["ap50_by_density"][b] = {"n_images": len(sub), "AP50": ap50_from_matches(m, sub) if sub else None}
        per_cls = defaultdict(list)
        for k, per in m["per_cat"].items():
            parts = list(per.values())
            npig = sum(p[3] for p in parts)
            if npig:
                per_cls[rarity[int(k)]].append(_ap_one_category(np.concatenate([p[0] for p in parts]),
                                                                np.concatenate([p[1] for p in parts]),
                                                                np.concatenate([p[2] for p in parts]), npig))
        out["ap50_by_rarity"] = {b: {"n_classes": len(v), "mean_AP50": float(np.mean(v)) if v else None} for b, v in per_cls.items()}
        res[cond], per_image[cond] = out, img_stats
        print(f"{cond}: done", flush=True)

    figs = [] if args.no_figures else make_figures(args, image_ids, rows, G, dens, per_image, name, store)
    payload = {"run_id": meta["run_id"], "source_run_git": meta["git_commit"], "split": split, "n_images": len(image_ids),
               "threshold": args.thr, "protocol": "standard", "rarity_cuts_n_images_dev": rarity_cuts,
               "conditions": res, "figures": figs}
    dst = REPO_ROOT / "outputs" / "metrics" / "m10_failure" / meta["run_id"]
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "failure.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    write_tables(payload, args.thr)
    return 0


def fmt_gt(s, keys=GT_OUTCOMES):
    return f"{s['n']} | " + " | ".join(f"{s[k]:.3f}" if s[k] is not None else "–" for k in keys)


def write_tables(p, thr):
    t = f"@{thr:g}"
    conds = list(p["conditions"])
    L = [f"# M10 failure analysis tables (generated by `scripts/m10_failure_analysis.py` from run `{p['run_id']}`)", "",
         f"{p['n_images']} dev images, standard protocol. Score threshold {thr:g} (D35) or `any` (all top-900 detections). "
         "Outcome definitions: `src/aerial_ovd/evaluation/failure.py`. Shares of GT objects; `detected` is coverage at IoU 0.5.", ""]
    for tag in ("any", t):
        L += [f"## GT outcomes by stratum ({tag})", ""]
        for cond in conds:
            g = p["conditions"][cond]["gt"][tag]
            L += [f"### {cond}", "", "| stratum | n GT | " + " | ".join(GT_OUTCOMES) + " |", "|---" * (len(GT_OUTCOMES) + 2) + "|"]
            order = (["all"] + [f"size:{x}" for x in SIZE_ORDER] + [f"density:{x}" for x in DENS_ORDER]
                     + [f"rarity:{x}" for x in RARE_ORDER] + sorted(k for k in g if k.startswith("source:")))
            L += [f"| {k} | {fmt_gt(g[k])} |" for k in order if k in g] + [""]
    L += [f"## Causes of detections with score ≥ {thr:g}", "", "| condition | stratum | n dets | " + " | ".join(DET_CAUSES) + " | FP of out-of-source class |",
          "|---" * (len(DET_CAUSES) + 4) + "|", ""][:-1]
    for cond in conds:
        for k, s in p["conditions"][cond]["det_causes"][t].items():
            L.append(f"| {cond} | {k} | {fmt_gt(s, DET_CAUSES)} | {s.get('fp_out_of_source', 0):.3f} |")
    L += ["", "## AP50 (maxDets 900) by objects per image", "", "| condition | " + " | ".join(DENS_ORDER) + " |", "|---" * 5 + "|"]
    for cond in conds:
        a = p["conditions"][cond]["ap50_by_density"]
        L.append(f"| {cond} | " + " | ".join(f"{a[b]['AP50']:.4f} (n={a[b]['n_images']})" if a[b]["AP50"] is not None else "–" for b in DENS_ORDER) + " |")
    L += ["", f"## Mean per-class AP50 by class rarity (terciles of dev images per class; cuts {p['rarity_cuts_n_images_dev']})", "",
          "| condition | " + " | ".join(RARE_ORDER) + " |", "|---" * 4 + "|"]
    for cond in conds:
        a = p["conditions"][cond]["ap50_by_rarity"]
        L.append(f"| {cond} | " + " | ".join(f"{a[b]['mean_AP50']:.4f} ({a[b]['n_classes']} cl.)" if b in a else "–" for b in RARE_ORDER) + " |")
    L += ["", f"## Most frequent confusions ({t}; GT class → class of the overlapping detection)", ""]
    for cond in conds:
        L += [f"**{cond}:** " + "; ".join(f"{c['gt']} → {c['predicted_as']} ({c['outcome']}, {c['n']})" for c in p["conditions"][cond]["confusions"][t][:8]), ""]
    if p["figures"]:
        L += ["## Qualitative examples (chosen by fixed rules, not by hand)", ""] + [f"- `{f['file']}`: {f['rule']}" for f in p["figures"]]
    out = REPO_ROOT / "reports" / "generated" / "M10_failure_tables.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"written: {out}")


def make_figures(args, image_ids, rows, G, dens, per_image, name, store):
    """Deterministic example selection on the headline condition (ties -> lowest image id)."""
    from PIL import Image, ImageDraw

    h = per_image.get(HEADLINE)
    if h is None:
        return []
    full = per_image.get("full", {})

    used = set()

    def pick(key, cond=lambda i: True):  # each figure uses a different image
        cands = [i for i in image_ids if cond(i) and i not in used]
        if not cands:
            return None
        best = min(cands, key=lambda i: (-key(i), i))
        used.add(best)
        return best

    n = {i: len(G[i][1]) for i in image_ids}
    rules = [
        ("tiny_objects", "most tiny GT objects not detected (headline, thr)", pick(lambda i: h[i]["tiny_missed"])),
        ("dense_scene", "most GT objects in the image (densest scene)", pick(lambda i: n[i])),
        ("similar_classes", "most wrong_class outcomes (headline, thr)", pick(lambda i: h[i]["wrong_class"])),
        ("rare_vocab_miss", "most rare-class GT objects dropped by the vocabulary (headline)", pick(lambda i: h[i]["rare_vocab_miss"])),
        ("selection_gain", "largest gain in true positives at thr, headline minus full", pick(lambda i: h[i]["tp_thr"] - full.get(i, Counter())["tp_thr"])),
        ("success", "highest detected share at thr among images with >= 3 GT", pick(lambda i: h[i]["detected"] / n[i], lambda i: n[i] >= 3)),
    ]
    out_dir = REPO_ROOT / "reports" / "figures" / "m10"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = []
    for tag, rule, iid in rules:
        if iid is None:
            continue
        panels = [HEADLINE] if tag != "selection_gain" else ["full", HEADLINE]
        ims = []
        for cond in panels:
            recs = read_records(args.run_dir / (cond.replace(":", "__") + ".jsonl"))
            r = recs[iid]
            with Image.open(store / "datasets" / "LAE-80C-images" / rows[iid]["file_name"]) as im:
                im = im.convert("RGB")
            s = 900 / max(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)))
            dr = ImageDraw.Draw(im)
            gb, gc = G[iid]
            for b in gb:
                dr.rectangle([b[0] * s, b[1] * s, (b[0] + b[2]) * s, (b[1] + b[3]) * s], outline=(0, 255, 0), width=2)
            d = np.array(r["detections"], float).reshape(-1, 6)
            cs, idx = det_causes(gb, gc, d[:, 2:6], d[:, 0].astype(int), d[:, 1], args.thr)
            for c, j in zip(cs, idx):
                b = d[j, 2:6]
                col = (0, 160, 255) if c == "tp" else (255, 40, 40)
                dr.rectangle([b[0] * s, b[1] * s, (b[0] + b[2]) * s, (b[1] + b[3]) * s], outline=col, width=2)
                if len(idx) <= 30:
                    dr.text((b[0] * s + 2, b[1] * s + 1), name[int(d[j, 0])], fill=col)
            dr.rectangle([0, 0, im.width, 16], fill=(0, 0, 0))
            dr.text((4, 2), f"{cond} | image {iid} ({rows[iid]['source']}) | green GT, blue TP, red FP (score >= {args.thr:g})", fill=(255, 255, 255))
            ims.append(im)
        canvas = Image.new("RGB", (sum(i.width for i in ims) + 10 * (len(ims) - 1), max(i.height for i in ims)), (255, 255, 255))
        x = 0
        for im in ims:
            canvas.paste(im, (x, 0))
            x += im.width + 10
        f = out_dir / f"{tag}.jpg"
        canvas.save(f, quality=75)
        figs.append({"file": f.relative_to(REPO_ROOT).as_posix(), "rule": rule, "image_id": iid, "source": rows[iid]["source"],
                     "n_gt": n[iid], "density": dens[iid], "panels": panels})
    return figs


if __name__ == "__main__":
    raise SystemExit(main())
