"""Module 6/7: uncertainty-aware retention and adaptive budgets vs fixed top-k (detector-free).

Protocol D32 (fixed before results):
  * calibration set = dev images NOT in the E0/M4/M5 200-image detection sample (530 images);
    evaluation set  = that 200-image sample (detection is evaluated on the same images later).
  * each rule's single parameter is calibrated on the calibration set to a TARGET MEAN SIZE m in {3, 8, 20}
    (never to maximise a metric); rules are compared on the evaluation set against fixed top-k at the
    same achieved mean size (top-k recall linearly interpolated over k).
  * conformal-style coverage check: threshold on p_g at the calibration quantile of GT-class probabilities for
    target pair coverage 1 - alpha, alpha in {0.5, 0.2, 0.1}; achieved coverage reported on evaluation images.
Signals: RemoteCLIP p_g (global softmax) and p_l (mean softmax over the 13 crops of grids 2x2+3x3) from
outputs/selection/dev/views_remoteclip_b32.npz (regenerate with scripts/run_views.py if absent).

Usage:
    python scripts/m6_retention.py
Writes outputs/metrics/m6_retention_dev/m6_metrics.json, reports/generated/M6_retention_tables.md,
outputs/figures/m6_recall_vs_size.png and outputs/selection/dev/m6_<rule>_m<m>.npz (boolean masks for detection).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aerial_ovd.utils.paths import get_store  # noqa: E402
from aerial_ovd.vocabulary.retention import calibrate, retain, threshold, top_p, topk  # noqa: E402
from aerial_ovd.vocabulary.selectors import softmax  # noqa: E402
from evaluate_selection import paired_mean_bootstrap  # noqa: E402
from run_detection import load_manifest  # noqa: E402

TARGETS = [3, 8, 20]
ALPHAS = [0.5, 0.2, 0.1]
E0_RUN = REPO_ROOT / "outputs" / "metrics" / "e0_headroom-e8585910cbfb" / "run_config.json"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    z = np.load(REPO_ROOT / "outputs" / "selection" / "dev" / "views_remoteclip_b32.npz")
    ids = [int(i) for i in z["image_ids"]]
    cat_ids = [int(c) for c in z["cat_ids"]]
    cidx = {c: j for j, c in enumerate(cat_ids)}
    mask13 = np.isin(z["crop_grid_ids"], [2, 3])
    PG = np.stack([softmax(g.astype(np.float64)) for g in z["global_logits"]])
    PL = np.stack([softmax(c[mask13].astype(np.float64), axis=-1).mean(0) for c in z["crop_logits"]])

    rows = {int(r["image_id"]): r for r in load_manifest() if r["split"] == "dev"}
    G = np.zeros(PG.shape, bool)
    for i, iid in enumerate(ids):
        for c in rows[iid]["classes"].split():
            G[i, cidx[int(c)]] = True
    gt = json.loads((get_store() / "datasets" / "LAE-80C-clean" / "dev.json").read_text(encoding="utf-8"))
    areas = defaultdict(list)
    for a in gt["annotations"]:
        areas[(a["image_id"], a["category_id"])].append(a["bbox"][2] * a["bbox"][3])
    small = np.zeros_like(G)
    for i, iid in enumerate(ids):
        for j in np.where(G[i])[0]:
            small[i, j] = float(np.median(areas[(iid, cat_ids[j])])) < 1024

    eval_ids = set(json.loads(E0_RUN.read_text(encoding="utf-8"))["image_ids"])
    ev = np.array([i in eval_ids for i in ids])
    ca = ~ev
    assert ev.sum() == 200 and ca.sum() == 530

    def apply(rule, param, idx):
        out = np.zeros((len(idx), PG.shape[1]), bool)
        for n, i in enumerate(idx):
            if rule == "topk":
                out[n] = topk(PG[i], int(param))
            elif rule == "threshold":
                out[n] = threshold(PG[i], param)
            elif rule == "top_p":
                out[n] = top_p(PG[i], param)
            elif rule.startswith("retain"):
                kb, d = param
                out[n] = retain(PG[i], PL[i], kb, d)
        return out

    def metrics(S, idx):
        g, sm = G[idx], small[idx]
        size = S.sum(1)
        rec = (S & g).sum(1) / g.sum(1)
        present = g.any(0)
        macro = float(np.mean([(S[:, j] & g[:, j]).sum() / g[:, j].sum() for j in np.where(present)[0]]))
        return {"mean_size": float(size.mean()), "size_sd": float(size.std()), "size_min": int(size.min()),
                "size_median": float(np.median(size)), "size_max": int(size.max()),
                "vocab_recall": float(rec.mean()), "macro_pair_recall": macro,
                "small_pair_recall": float(S[g & sm].mean()), "full_coverage": float(((S & g).sum(1) == g.sum(1)).mean()),
                "_rec": rec, "_small_hits": (S & g & sm).sum(1), "_small_n": (g & sm).sum(1)}

    ca_idx, ev_idx = np.where(ca)[0], np.where(ev)[0]
    # fixed top-k curve on evaluation images (for matched-size interpolation)
    curve = {k: metrics(apply("topk", k, ev_idx), ev_idx) for k in range(1, 41)}

    def interp(size, key):
        ks = np.arange(1, 41)
        return float(np.interp(size, ks, [curve[k][key] for k in ks]))

    out = {"protocol": "D32", "n_calibration": int(ca.sum()), "n_evaluation": int(ev.sum()),
           "signals": "RemoteCLIP p_g global softmax; p_l mean softmax over 13 crops (2x2+3x3)", "targets": {}}
    exports = {}
    for m in TARGETS:
        res = {}
        rules = {
            "topk": (m, None),
            "threshold": (calibrate(lambda t: apply("threshold", t, ca_idx).sum(1).mean(), m, 0.0, 1.0, increasing=False), None),
            "top_p": (calibrate(lambda t: apply("top_p", t, ca_idx).sum(1).mean(), m, 0.0, 1.0, increasing=True), None),
        }
        kb = max(1, round(m / 2))
        d = calibrate(lambda dd: apply("retain", (kb, dd), ca_idx).sum(1).mean(), m, -1.0, 1.0, increasing=False)
        rules["retain"] = ((kb, d), None)
        for name, (param, _) in rules.items():
            S_ev = apply(name, param, ev_idx)
            mt = metrics(S_ev, ev_idx)
            mt["calibration_mean_size"] = float(apply(name, param, ca_idx).sum(1).mean())
            mt["param"] = param if not isinstance(param, tuple) else {"k_base": param[0], "d": param[1]}
            mt["topk_interp_at_same_size"] = {k: interp(mt["mean_size"], k) for k in ("vocab_recall", "macro_pair_recall", "small_pair_recall")}
            mt["advantage_vs_topk_same_size"] = {k: mt[k] - mt["topk_interp_at_same_size"][k] for k in ("vocab_recall", "macro_pair_recall", "small_pair_recall")}
            if name != "topk":
                base = metrics(apply("topk", m, ev_idx), ev_idx)
                mt["bootstrap_vs_topk_m"] = {"vocab_recall": paired_mean_bootstrap(base["_rec"], mt["_rec"])}
            res[name] = mt
            if name != "topk":
                exports[f"m6_{name}_m{m}"] = (apply(name, param, np.arange(len(ids))), mt["param"], m, mt)
        out["targets"][str(m)] = res
    # conformal-style coverage check
    cov = {}
    calib_probs = PG[ca_idx][G[ca_idx]]
    for a in ALPHAS:
        t = float(np.quantile(calib_probs, a))  # keep classes with p_g >= t -> ~(1 - a) of calibration GT pairs
        S = apply("threshold", t, ev_idx)
        g = G[ev_idx]
        cov[str(a)] = {"threshold": t, "target_pair_coverage": 1 - a, "achieved_pair_coverage": float(S[g].mean()),
                       "mean_size": float(S.sum(1).mean()), "size_max": int(S.sum(1).max()),
                       "note": "sets capped at 40 classes; coverage is marginal over (image, class) pairs"}
    out["conformal_coverage"] = cov

    sel_dir = REPO_ROOT / "outputs" / "selection" / "dev"
    for name, (mask, param, m, mt) in exports.items():
        np.savez_compressed(sel_dir / f"{name}.npz", image_ids=np.array(ids), cat_ids=np.array(cat_ids), selected=mask,
                            meta=json.dumps({"selector": name, "param": param, "target_mean_size": m, "protocol": "D32",
                                             "calibration_images": int(ca.sum()), "views": "views_remoteclip_b32.npz",
                                             "views_git_commit": json.loads(str(z["meta"]))["git_commit"]}))
    for m in out["targets"].values():
        for r in m.values():
            for k in [k for k in r if k.startswith("_")]:
                r.pop(k)
    mdir = REPO_ROOT / "outputs" / "metrics" / "m6_retention_dev"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "m6_metrics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")

    L = ["# M6/M7 retention & adaptive budgets (generated by `scripts/m6_retention.py`)", "",
         f"Protocol D32: parameters calibrated on {out['n_calibration']} dev images to a target mean size; "
         f"all numbers below are on the {out['n_evaluation']} held-out evaluation images (the detection sample). "
         "`vs top-k` = difference to fixed top-k at the same achieved mean size (interpolated). Base signal: RemoteCLIP.", "",
         "| target | rule | param | mean size (sd; min–max) | image vocab recall (vs top-k) | macro pair recall (vs top-k) | small-pair recall (vs top-k) | full coverage | recall Δ vs top-k=m, 95% CI |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m, res in out["targets"].items():
        for name, r in res.items():
            adv = r["advantage_vs_topk_same_size"]
            b = r.get("bootstrap_vs_topk_m", {}).get("vocab_recall")
            bs = f"{b['diff']:+.3f} [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}]" + ("*" if b["excludes_zero"] else "") if b else "ref"
            p = r["param"]
            ps = f"k={p}" if name == "topk" else (f"k_base={p['k_base']}, d={p['d']:.3f}" if isinstance(p, dict) else f"{p:.4f}")
            L.append(f"| {m} | {name} | {ps} | {r['mean_size']:.2f} ({r['size_sd']:.1f}; {r['size_min']}–{r['size_max']}) "
                     f"| {r['vocab_recall']:.3f} ({adv['vocab_recall']:+.3f}) | {r['macro_pair_recall']:.3f} ({adv['macro_pair_recall']:+.3f}) "
                     f"| {r['small_pair_recall']:.3f} ({adv['small_pair_recall']:+.3f}) | {r['full_coverage']:.3f} | {bs} |")
    L += ["", "## Conformal-style coverage check (threshold calibrated on calibration images)", "",
          "| target pair coverage | achieved on evaluation images | mean size | max size |", "|---|---|---|---|"]
    for a, c in cov.items():
        L.append(f"| {c['target_pair_coverage']:.2f} | {c['achieved_pair_coverage']:.3f} | {c['mean_size']:.2f} | {c['size_max']} |")
    gen = REPO_ROOT / "reports" / "generated" / "M6_retention_tables.md"
    gen.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ks = np.arange(1, 41)
    for ax, key, lab in ((axes[0], "macro_pair_recall", "macro (class-averaged) pair recall"),
                         (axes[1], "small_pair_recall", "small-object pair recall")):
        ax.plot(ks, [curve[k][key] for k in ks], color="gray", label="fixed top-k")
        for name, mk in (("threshold", "o"), ("top_p", "s"), ("retain", "^")):
            xs = [out["targets"][str(m)][name]["mean_size"] for m in TARGETS]
            ys = [out["targets"][str(m)][name][key] for m in TARGETS]
            ax.scatter(xs, ys, marker=mk, label=name, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("mean vocabulary size (evaluation images)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("M6/M7: variable-size rules vs fixed top-k at matched size (200 held-out dev images)")
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "outputs" / "figures" / "m6_recall_vs_size.png", dpi=150)
    print(gen.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
