"""E9 (D42): synonym-aware vocabulary selection, motivated by M10 finding 2 (cross-source near-synonym classes).

Method (deployable; class names only): headline rule (RemoteCLIP global p_g >= t, D34) -> add every class whose
class-name text similarity to a selected class is >= tau (RemoteCLIP text encoder, the selector's own templates).
Protocol (pre-declared, D42), dev only, the 200 detection images are NEVER used for any choice:
  * tau grid: 1.0 (= no expansion = D34), 0.95, 0.925, 0.90, 0.875, 0.85, 0.825, 0.80, 0.775, 0.75
  * for each tau, the base threshold t is calibrated on the 530 calibration images to a final mean size of 8 (D32)
  * tau values whose calibrated mean size misses 8 by > 0.5 are ineligible (amendment D42a)
  * tau chosen on the calibration images by J = (macro pair recall + object-weighted coverage) / 2; ties -> larger tau
Also written: the chosen tau with the unchanged D34 threshold (size grows), and a DIAGNOSTIC that uses source
identity (threshold within the image's source label space, calibrated to mean size 8) -- an upper bound, not a method.
Usage:
    python scripts/e9_synonym_selector.py
Writes outputs/selection/dev/{e9_syn_m8,e9_syn_d34,e9_diag_insource_m8}.npz, outputs/metrics/e9_synonym_dev/e9.json,
reports/generated/E9_synonym_tables.md.
"""

from __future__ import annotations

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
from aerial_ovd.vocabulary.names import prompt_name  # noqa: E402
from aerial_ovd.vocabulary.retention import calibrate, expand_synonyms, threshold  # noqa: E402
from aerial_ovd.vocabulary.selectors import DEFAULT_TEMPLATES, softmax  # noqa: E402
from evaluate_selection import paired_mean_bootstrap  # noqa: E402
from run_detection import load_manifest  # noqa: E402

TAUS = [1.0, 0.95, 0.925, 0.90, 0.875, 0.85, 0.825, 0.80, 0.775, 0.75]
TARGET = 8
D34_T = 0.023681640625
E0_RUN = REPO_ROOT / "outputs" / "metrics" / "e0_headroom-e8585910cbfb" / "run_config.json"


def text_similarity(names: list[str]) -> np.ndarray:
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32")
    sd = torch.load(get_store() / "checkpoints" / "remoteclip-vit-b-32" / "RemoteCLIP-ViT-B-32.pt", map_location="cpu")
    model.load_state_dict(sd, strict=True)  # the "no pretrained weights" warning above is expected
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    embs = []
    with torch.no_grad():
        for n in names:  # same template averaging as CLIPSelector._encode_text
            e = model.encode_text(tok([t.format(n) for t in DEFAULT_TEMPLATES])).float()
            e = (e / e.norm(dim=-1, keepdim=True)).mean(0)
            embs.append((e / e.norm()).numpy())
    E = np.stack(embs)
    return E @ E.T


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    z = np.load(REPO_ROOT / "outputs" / "selection" / "dev" / "views_remoteclip_b32.npz")
    ids = [int(i) for i in z["image_ids"]]
    cat_ids = [int(c) for c in z["cat_ids"]]
    PG = np.stack([softmax(g.astype(np.float64)) for g in z["global_logits"]])
    vocab = list(csv.DictReader((REPO_ROOT / "data/manifests/lae80c_vocabulary.csv").open(encoding="utf-8")))
    assert [int(v["category_id"]) for v in vocab] == cat_ids
    names = [prompt_name(v["raw_name"], "light") for v in vocab]
    cat_src = np.array([v["source"] for v in vocab])
    rows = {int(r["image_id"]): r for r in load_manifest() if r["split"] == "dev"}
    img_src = np.array([rows[i]["source"] for i in ids])
    cidx = {c: j for j, c in enumerate(cat_ids)}
    N = np.zeros(PG.shape)  # GT object counts per (image, class)
    gt = json.loads((get_store() / "datasets" / "LAE-80C-clean" / "dev.json").read_text(encoding="utf-8"))
    row = {i: k for k, i in enumerate(ids)}
    for a in gt["annotations"]:
        N[row[a["image_id"]], cidx[a["category_id"]]] += 1
    G = N > 0
    eval_ids = set(json.loads(E0_RUN.read_text(encoding="utf-8"))["image_ids"])
    ev = np.array([i in eval_ids for i in ids])
    ca = ~ev
    assert ev.sum() == 200 and ca.sum() == 530
    ca_idx, ev_idx = np.where(ca)[0], np.where(ev)[0]

    S = text_similarity(names)

    def masks(t, tau, idx, in_source=False):
        out = []
        for i in idx:
            p = np.where(cat_src == img_src[i], PG[i], 0.0) if in_source else PG[i]
            out.append(expand_synonyms(threshold(p, t), S, tau, p=PG[i]))
        return np.stack(out)

    def metrics(M, idx):
        g, n = G[idx], N[idx]
        present = np.where(g.any(0))[0]
        rec = (M & g).sum(1) / g.sum(1)
        return {"mean_size": float(M.sum(1).mean()), "macro_pair_recall": float(np.mean([(M[:, j] & g[:, j]).sum() / g[:, j].sum() for j in present])),
                "object_weighted": float((n * M).sum() / n.sum()), "image_mean_recall": float(rec.mean()),
                "out_of_source_share": float(np.mean([(M[k] & (cat_src != img_src[i])).sum() / M[k].sum() for k, i in enumerate(idx)])),
                "_rec": rec}

    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731
    grid = []
    for tau in TAUS:
        t = calibrate(lambda tt: masks(tt, tau, ca_idx).sum(1).mean(), TARGET, 0.0, 1.0, increasing=False)
        r = metrics(masks(t, tau, ca_idx), ca_idx)
        r.update({"tau": tau, "t": t, "J": (r["macro_pair_recall"] + r["object_weighted"]) / 2})
        grid.append(r)
        print(f"tau={tau:<6} t={t:.4f} size={r['mean_size']:.2f} macro={r['macro_pair_recall']:.3f} obj={r['object_weighted']:.3f} J={r['J']:.3f}", flush=True)
    for r in grid:  # matched size is required (D32/D42): tau values that cannot reach the target are ineligible
        r["feasible"] = abs(r["mean_size"] - TARGET) <= 0.5
    best = max((r for r in grid if r["feasible"]), key=lambda r: (round(r["J"], 6), r["tau"]))
    tau, t8 = best["tau"], best["t"]
    t_diag = calibrate(lambda tt: masks(tt, 1.0, ca_idx, in_source=True).sum(1).mean(), TARGET, 0.0, 1.0, increasing=False)

    all_idx = np.arange(len(ids))
    sets = {"e9_syn_m8": (masks(t8, tau, all_idx), {"tau": tau, "t": t8, "rule": "threshold t (calibrated to mean size 8 after expansion) + synonym expansion"}),
            "e9_syn_d34": (masks(D34_T, tau, all_idx), {"tau": tau, "t": D34_T, "rule": "D34 threshold unchanged + synonym expansion"}),
            "e9_diag_insource_m8": (masks(t_diag, 1.0, all_idx, in_source=True), {"t": t_diag, "rule": "DIAGNOSTIC: threshold within the image's source label space (uses source identity)"})}
    headline = np.load(REPO_ROOT / "outputs/selection/dev/m6_threshold_m8.npz")["selected"]
    assert (masks(D34_T, 1.0, all_idx) == headline).all(), "D34 headline not reproduced"
    out = {"protocol": "D42", "taus": TAUS, "target_mean_size": TARGET, "n_calibration": int(ca.sum()), "n_evaluation": int(ev.sum()),
           "calibration_grid": [strip(r) for r in grid], "chosen_tau": tau,
           "eval": {"headline (D34)": strip(metrics(headline[ev_idx], ev_idx))}, "eval_vs_headline_image_recall": {}}
    h = metrics(headline[ev_idx], ev_idx)
    for name, (M, meta) in sets.items():
        np.savez_compressed(REPO_ROOT / f"outputs/selection/dev/{name}.npz", image_ids=np.array(ids), cat_ids=np.array(cat_ids), selected=M,
                            meta=json.dumps({"selector": name, "protocol": "D42", "calibration_images": int(ca.sum()), **meta}))
        r = metrics(M[ev_idx], ev_idx)
        out["eval"][name] = strip(r)
        out["eval_vs_headline_image_recall"][name] = paired_mean_bootstrap(h["_rec"], r["_rec"])
    # which synonym pairs does the chosen tau link?
    iu = np.triu_indices(len(names), 1)
    pairs = sorted(((float(S[a, b]), names[a], cat_src[a], names[b], cat_src[b]) for a, b in zip(*iu) if S[a, b] >= tau), reverse=True)
    out["linked_pairs"] = [{"sim": s, "a": f"{a} [{sa}]", "b": f"{b} [{sb}]"} for s, a, sa, b, sb in pairs]
    dst = REPO_ROOT / "outputs/metrics/e9_synonym_dev"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "e9.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")

    L = ["# E9 synonym-aware selection (generated by `scripts/e9_synonym_selector.py`; protocol D42)", "",
         f"## Calibration ({out['n_calibration']} dev images; used for every choice)", "",
         "| tau | t | mean size | macro pair recall | object-weighted | J |", "|---|---|---|---|---|---|"]
    L += [f"| {r['tau']} | {r['t']:.4f} | {r['mean_size']:.2f} | {r['macro_pair_recall']:.3f} | {r['object_weighted']:.3f} | {r['J']:.3f}"
          + (" ← chosen" if r["tau"] == tau else "" if r["feasible"] else " (cannot reach size 8: ineligible)") + " |" for r in grid]
    L += ["", f"## Held-out evaluation ({out['n_evaluation']} dev images; reported, never used for choices)", "",
          "| selection | mean size | macro pair recall | object-weighted | image recall | out-of-source share | image recall vs headline (95% CI) |", "|---|---|---|---|---|---|---|"]
    for name, r in out["eval"].items():
        d = out["eval_vs_headline_image_recall"].get(name)
        ds = f"{d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]" + ("*" if d["excludes_zero"] else "") if d else "ref"
        L.append(f"| {name} | {r['mean_size']:.2f} | {r['macro_pair_recall']:.3f} | {r['object_weighted']:.3f} | {r['image_mean_recall']:.3f} | {r['out_of_source_share']:.2f} | {ds} |")
    L += ["", f"## Class pairs linked at tau = {tau} ({len(pairs)})", ""] + [f"- {p['sim']:.3f}: {p['a']} ↔ {p['b']}" for p in out["linked_pairs"][:60]]
    (REPO_ROOT / "reports/generated/E9_synonym_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
