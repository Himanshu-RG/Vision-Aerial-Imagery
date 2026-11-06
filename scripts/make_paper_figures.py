"""Paper figures for the CVPR draft, built only from committed metrics, selection files and local images.

Writes into reports/paper/cvpr_draft/fig/:
  pipeline.pdf        method overview on two real dev images (RemoteCLIP probabilities, kept classes, detections)
  dev_size_curve.pdf  dev AP50 vs mean vocabulary size for fixed top-k selectors and the adaptive rules
  test_analysis.pdf   test split: per-source AP50, vocabulary-size distribution, per-class AP50 change
  qual_full.jpg, qual_ours.jpg, qual_fail.jpg   qualitative panels (M10 figures with the run header cropped)

Usage:  python scripts/make_paper_figures.py      (image store from AOVD_STORE or configs/paths.local.yaml)
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/paper/cvpr_draft/fig"
MET = ROOT / "outputs/metrics"
SEL = ROOT / "outputs/selection"

T_HEADLINE = 0.023681640625      # D34 threshold, calibrated on 530 dev images
LOGIT_SCALE = 99.85401153564453  # RemoteCLIP ViT-B/32 exp(logit_scale)
SCORE_THR = 0.14                 # D35 operating point used for drawing boxes

C_FULL, C_OURS, C_ORACLE = "#8a8a8a", "#1f5aa6", "#e39b2d"
C_RC, C_CLIP, C_FREQ, C_RAND = "#2a9d8f", "#d1495b", "#8d6a3f", "#b9b9b9"
SRC_COL = {"DIOR-18C": "#1f5aa6", "DOTA-6C": "#5fa8d3", "FAIR-18C": "#d1495b", "Xview-34C": "#e39b2d", "PP-4C": "#6a994e"}
SRC_NAME = {"DIOR-18C": "DIOR", "DOTA-6C": "DOTA", "FAIR-18C": "FAIR1M", "Xview-34C": "xView", "PP-4C": "PP"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
})


def store() -> Path:
    if os.environ.get("AOVD_STORE"):
        return Path(os.environ["AOVD_STORE"])
    cfg = ROOT / "configs/paths.local.yaml"
    for line in cfg.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("store:"):
            return Path(line.split(":", 1)[1].split("#")[0].strip().strip('"').strip("'"))
    raise SystemExit("set AOVD_STORE")


def metrics(run: str) -> dict:
    return json.loads((MET / run / "metrics.json").read_text(encoding="utf-8"))


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


# ---------------------------------------------------------------- figure 1: pipeline
def fig_pipeline(st: Path) -> None:
    dev = json.loads((st / "datasets/LAE-80C-clean/dev.json").read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in dev["categories"]}
    ims = {i["id"]: i for i in dev["images"]}
    z = np.load(SEL / "dev/remoteclip_b32.npz")
    ids, cid = list(z["image_ids"]), list(z["cat_ids"])
    mk = np.load(SEL / "dev/m6_threshold_m8.npz")
    mids = list(mk["image_ids"])
    pred_dir = ROOT / "outputs/predictions/e5_caption_tagclip_det-97b9c543162f"
    preds = {}
    for line in (pred_dir / "set__m6_threshold_m8.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        preds[r["image_id"]] = r

    examples = [596, 2160]  # the two dev images whose headline predictions are stored locally
    W, H = 6.9, 2.75
    fig = plt.figure(figsize=(W, H))
    side = 0.98 / H                      # square image panels, 0.98 in high
    col_x = {"img": 0.006, "bar": 0.305, "prm": 0.535, "gd": 0.695, "out": 0.995 - 0.98 / W}
    bar_w, prm_w, gd_w = 0.17, 0.125, 0.085
    rows_y = [0.505, 0.06]
    heads = {"img": "Input image", "bar": "RemoteCLIP class probabilities", "prm": "Prompt",
             "out": "Detections"}

    def arrow(x0, x1, y):
        fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=7, lw=0.8, color="#444444"))

    for row, iid in enumerate(examples):
        y0 = rows_y[row]
        yc = y0 + side / 2
        info = ims[iid]
        img = Image.open(st / "datasets/LAE-80C-images" / info["file_name"]).convert("RGB")
        p = softmax(LOGIT_SCALE * z["scores"][ids.index(iid)].astype(np.float64))
        keep = mk["selected"][mids.index(iid)]
        kept_names = [cats[cid[j]] for j in np.argsort(-p) if keep[j]]
        gts = [a for a in dev["annotations"] if a["image_id"] == iid]
        src = SRC_NAME[info["file_name"].split("_")[0]]

        ax0 = fig.add_axes([col_x["img"], y0, 0.98 / W, side])
        ax0.imshow(img, aspect="auto")
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.text(0.02, 0.98, f"{src}", transform=ax0.transAxes, ha="left", va="top", fontsize=6, color="w",
                 weight="bold", bbox=dict(fc="k", alpha=0.45, lw=0, pad=1))

        ax1 = fig.add_axes([col_x["bar"], y0 + 0.07, bar_w, side - 0.07])
        order = np.argsort(-p)[:8][::-1]
        cols = [C_OURS if keep[j] else "#c4c4c4" for j in order]
        ax1.barh(range(len(order)), p[order], color=cols, height=0.72)
        ax1.set_yticks(range(len(order)))
        ax1.set_yticklabels([cats[cid[j]] for j in order], fontsize=5.6)
        ax1.set_xscale("log")
        ax1.set_xlim(1e-4, 1.5)
        ax1.set_xticks([1e-4, 1e-2, 1])
        ax1.minorticks_off()
        ax1.axvline(T_HEADLINE, color="k", ls="--", lw=0.7)
        ax1.text(T_HEADLINE * 1.25, len(order) - 0.45, "$t$", fontsize=6.2, va="bottom")
        ax1.tick_params(axis="y", length=0, pad=1.5)
        ax1.tick_params(axis="x", labelsize=5.6, pad=1)
        if row == 1:
            ax1.set_xlabel("$p(c\\mid I)$, log scale (top 8 of 80)", labelpad=1, fontsize=5.8)

        ax2 = fig.add_axes([col_x["prm"], y0 + 0.05, prm_w, side - 0.1])
        ax2.axis("off")
        shown = kept_names if len(kept_names) <= 4 else kept_names[:4] + ["..."]
        txt = " .\n".join(shown) + (" ." if shown[-1] != "..." else "")
        ax2.add_patch(FancyBboxPatch((0.0, 0.16), 1.0, 0.8, boxstyle="round,pad=0.02", fc="#eef3fb", ec=C_OURS,
                                     lw=0.8, transform=ax2.transAxes, clip_on=False))
        ax2.text(0.5, 0.56, txt, ha="center", va="center", fontsize=5.8, transform=ax2.transAxes,
                 family="DejaVu Sans Mono")
        ax2.text(0.5, 0.0, f"{len(kept_names)} of 80 classes", ha="center", va="bottom", fontsize=6,
                 transform=ax2.transAxes, color=C_OURS, weight="bold")

        ax4 = fig.add_axes([col_x["out"], y0, 0.98 / W, side])
        ax4.imshow(img, aspect="auto")
        ax4.set_xticks([]); ax4.set_yticks([])
        for a in gts:
            x, y, w, h = a["bbox"]
            ax4.add_patch(Rectangle((x, y), w, h, fill=False, ec="#2bd92b", lw=0.5))
        dets = sorted([d for d in preds[iid]["detections"] if d[1] >= SCORE_THR], key=lambda d: -d[1])
        used, n_tp = set(), 0
        for d in dets:
            best, bj = 0.0, None
            for j, a in enumerate(gts):
                if j in used or a["category_id"] != d[0]:
                    continue
                v = iou(d[2:6], a["bbox"])
                if v > best:
                    best, bj = v, j
            tp = best >= 0.5
            if tp:
                used.add(bj)
                n_tp += 1
            ax4.add_patch(Rectangle((d[2], d[3]), d[4], d[5], fill=False, ec="#1e90ff" if tp else "#ff3030", lw=0.6))
        ax4.text(0.02, 0.98, f"{len(gts)} GT, {n_tp} TP", transform=ax4.transAxes, ha="left", va="top", fontsize=6,
                 color="w", weight="bold", bbox=dict(fc="k", alpha=0.45, lw=0, pad=1))

        arrow(col_x["img"] + 0.98 / W + 0.004, col_x["bar"] - 0.118, yc)
        arrow(col_x["bar"] + bar_w + 0.008, col_x["prm"] - 0.006, yc)
        arrow(col_x["prm"] + prm_w + 0.006, col_x["gd"] - 0.004, yc)
        arrow(col_x["gd"] + gd_w + 0.004, col_x["out"] - 0.006, yc)

    top = rows_y[0] + side
    for k, h in heads.items():
        x = {"img": col_x["img"] + 0.49 / W, "bar": col_x["bar"] - 0.03 + bar_w / 2,
             "prm": col_x["prm"] + prm_w / 2, "out": col_x["out"] + 0.49 / W}[k]
        fig.text(x, top + 0.025, h, ha="center", va="bottom", fontsize=6.8, weight="bold")
    fig.add_artist(FancyBboxPatch((col_x["gd"], rows_y[1] + 0.04), gd_w, top - rows_y[1] - 0.08,
                                  boxstyle="round,pad=0.006", fc="#f4f4f4", ec="#555555", lw=0.8,
                                  transform=fig.transFigure))
    fig.text(col_x["gd"] + gd_w / 2, (rows_y[1] + top) / 2, "Grounding\nDINO-T\n\nfrozen,\nno training",
             ha="center", va="center", fontsize=6.4, weight="bold")
    fig.savefig(OUT / "pipeline.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2: dev AP vs vocabulary size
def fig_dev_curve() -> None:
    m4 = metrics("m4_selectors_det-16d38bd056f8")["conditions"]
    m6 = metrics("m6_retention_det-8febbe4deb4a")["conditions"]

    def pts(conds, keys):
        return [conds[k]["vocab_size_mean"] for k in keys], {pr: [conds[k][pr]["AP50@900"] for k in keys]
                                                             for pr in ("standard", "source_aware")}

    fixed = {
        "RemoteCLIP top-$k$": ([f"sel:remoteclip_b32:k{k}" for k in (3, 8, 20, 40)], C_RC, "o", m4),
        "OpenAI CLIP top-$k$": ([f"sel:clip_openai_b32:k{k}" for k in (3, 8, 20, 40)], C_CLIP, "s", m4),
        "Frequency prior top-$k$": ([f"sel:frequency:k{k}" for k in (3, 8, 20, 40)], C_FREQ, "^", m4),
        "Random top-$k$": ([f"sel:random_s0:k{k}" for k in (3, 8, 20)], C_RAND, "v", m4),
        "Adaptive top-$p$": ([f"set:m6_top_p_m{k}" for k in (3, 8, 20)], "#7b5ea7", "D", m6),
        "Adaptive threshold (ours)": ([f"set:m6_threshold_m{k}" for k in (3, 8, 20)], C_OURS, "*", m6),
    }
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.75), sharey=False)
    for ax, pr, title in zip(axes, ("standard", "source_aware"), ("Standard", "Source-aware")):
        for name, (keys, col, mkr, conds) in fixed.items():
            x, y = pts(conds, keys)
            ms = 6 if mkr == "*" else 3
            ax.plot(x, y[pr], marker=mkr, color=col, lw=1.0 if mkr == "*" else 0.8, ms=ms, label=name,
                    zorder=3 if mkr == "*" else 2)
        ax.scatter([80], [m4["full"][pr]["AP50@900"]], marker="X", color=C_FULL, s=18, zorder=4, label="All 80 classes")
        ax.set_xscale("log")
        ax.set_xticks([3, 8, 20, 40, 80])
        ax.set_xticklabels(["3", "8", "20", "40", "80"])
        ax.minorticks_off()
        ax.set_xlabel("mean classes per image")
        ax.set_title(f"{title} (oracle {m4['oracle_gt'][pr]['AP50@900']:.3f})", pad=2, fontsize=6.6)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("AP50 on dev")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=1.6,
               columnspacing=0.8, fontsize=5.6)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.9, bottom=0.42, wspace=0.34)
    fig.savefig(OUT / "dev_size_curve.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3: test analysis
def fig_test() -> None:
    run = "final_test-75ab843910a3"
    ps = json.loads((MET / "m11_per_source" / f"{run}.json").read_text(encoding="utf-8"))["per_source"]
    m = metrics(run)["conditions"]
    vocab = {r["category_id"]: r for r in csv.DictReader((ROOT / "data/manifests/lae80c_vocabulary.csv").open(encoding="utf-8"))}
    manifest = {r["image_id"]: r for r in csv.DictReader((ROOT / "data/manifests/lae80c_manifest.csv").open(encoding="utf-8"))}

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.9), gridspec_kw={"width_ratios": [1.15, 1.0, 1.55]})

    # (a) per-source source-aware AP50
    ax = axes[0]
    srcs = ["DIOR-18C", "DOTA-6C", "FAIR-18C", "Xview-34C", "PP-4C"]
    w = 0.26
    for k, (cond, col, lab) in enumerate((("full", C_FULL, "All 80 classes"), ("set:headline_d34", C_OURS, "Ours"),
                                          ("oracle_gt", C_ORACLE, "Oracle (GT classes)"))):
        vals = [ps[s]["conditions"][cond]["source_aware"]["AP50"] for s in srcs]
        ax.bar(np.arange(len(srcs)) + (k - 1) * w, vals, w, color=col, label=lab)
    ax.set_xticks(range(len(srcs)))
    ax.set_xticklabels([f"{SRC_NAME[s]}\n({ps[s]['n_images']})" for s in srcs], fontsize=6)
    ax.axvspan(-0.5, 1.5, color="#1f5aa6", alpha=0.06, lw=0)
    ax.text(0.5, 0.186, "seen by\nRemoteCLIP", ha="center", va="center", fontsize=5.6, color="#1f5aa6")
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, 0.2)
    ax.set_ylabel("source-aware AP50 (test)")
    ax.set_title("(a) Per source dataset", pad=2)
    ax.legend(frameon=False, loc="upper right", fontsize=5.6, bbox_to_anchor=(1.02, 1.0), handlelength=1.2)

    # (b) vocabulary size distribution of the frozen rule on test
    ax = axes[1]
    z = np.load(SEL / "test/headline_d34.npz")
    sizes = z["selected"].sum(1)
    src_of = np.array([manifest[str(i)]["source"] for i in z["image_ids"]])
    seen = np.isin(src_of, ["DIOR-18C", "DOTA-6C"])
    bins = np.arange(0.5, sizes.max() + 1.5, 1)
    ax.hist([sizes[seen], sizes[~seen]], bins=bins, stacked=True, color=["#1f5aa6", "#d1495b"],
            label=[f"seen sources ({seen.sum()})", f"unseen sources ({(~seen).sum()})"], rwidth=0.9)
    ax.axvline(sizes.mean(), color="k", ls="--", lw=0.7, label=f"mean = {sizes.mean():.1f}")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.36)
    ax.set_xlabel("classes kept per image")
    ax.set_ylabel("test images")
    ax.set_title("(b) Adaptive vocabulary size", pad=2)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="none", fontsize=5.4, loc="upper left", bbox_to_anchor=(-0.02, 1.02), handlelength=1.0)

    # (c) per-class change in source-aware AP50, ours - full
    ax = axes[2]
    pf = m["full"]["source_aware"]["per_class"]
    po = m["set:headline_d34"]["source_aware"]["per_class"]
    keys = list(pf)  # category ids as strings; all 80 classes have test GT
    d = sorted(((po[k]["AP50"] - pf[k]["AP50"], k) for k in keys), reverse=True)
    vals = [v for v, _ in d]
    cols = [SRC_COL[vocab[k]["source"]] for _, k in d]
    ax.bar(range(len(vals)), vals, color=cols, width=0.85)
    ax.axhline(0, color="k", lw=0.5)
    better = sum(v > 1e-12 for v in vals)
    worse = sum(v < -1e-12 for v in vals)
    ax.text(0.98, 0.95, f"{better} classes up, {worse} down,\n{len(vals) - better - worse} unchanged",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.8)
    ax.set_xticks([])
    ax.set_xlabel(f"{len(vals)} classes, sorted by change")
    ax.set_ylabel("$\\Delta$ AP50 (ours $-$ all 80)")
    ax.set_title("(c) Per-class change (source-aware)", pad=2)
    handles = [Rectangle((0, 0), 1, 1, color=SRC_COL[s]) for s in srcs]
    ax.legend(handles, [SRC_NAME[s] for s in srcs], frameon=False, ncol=5, fontsize=5.4, loc="lower left",
              handlelength=0.9, columnspacing=0.6, bbox_to_anchor=(0.0, -0.02))
    fig.subplots_adjust(left=0.065, right=0.995, top=0.9, bottom=0.2, wspace=0.33)
    fig.savefig(OUT / "test_analysis.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- qualitative panels
def fig_qual() -> None:
    src = ROOT / "reports/figures/m10"
    g = Image.open(src / "selection_gain.jpg").convert("RGB")
    head = 17  # height of the run-label strip at the top of the M10 renders
    g.crop((0, head, 900, 900)).save(OUT / "qual_full.jpg", quality=92)
    g.crop((910, head, 1810, 900)).save(OUT / "qual_ours.jpg", quality=92)
    Image.open(src / "similar_classes.jpg").convert("RGB").crop((0, head, 900, 900)).save(OUT / "qual_fail.jpg", quality=92)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    st = store()
    fig_pipeline(st)
    fig_dev_curve()
    fig_test()
    fig_qual()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
