"""E5: VocAda-style caption -> class selector (and the TagCLIP-style crop-level export), detector-free.

VocAda-style (NOT a reproduction; D38): Florence-2-base detailed caption (VocAda used LLaVA-Next-7B) ->
candidate phrases = all 1-3-grams of consecutive non-stopword tokens (VocAda used spaCy noun phrases) ->
class score = max over phrases of CLIP ViT-B/32 text-text cosine(phrase, class name) (VocAda: CLIP ViT-L/14
text similarity, top-k per noun). Here the per-class max score supports both top-k and a threshold calibrated on
the 530 calibration images to a target mean size (D32), so it is compared at matched budgets.

TagCLIP-style (crop-level analogue, NOT a reproduction of TagCLIP's patch-level method): OpenAI CLIP local-only,
max over all 29 crops (M5 variant a0_max_g234).

Usage:
    python scripts/e5_caption_selector.py
Writes outputs/selection/dev/{caption_clip.npz, e5_caption_thr_m3.npz, e5_caption_thr_m8.npz, tagclip_style.npz},
outputs/metrics/e5_selection_dev/e5_metrics.json and reports/generated/E5_selection_tables.md.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aerial_ovd.utils.paths import get_store  # noqa: E402
from aerial_ovd.vocabulary.names import prompt_name  # noqa: E402
from aerial_ovd.vocabulary.retention import calibrate, threshold, topk  # noqa: E402
from aerial_ovd.vocabulary.selectors import combine_views  # noqa: E402
from evaluate_selection import paired_mean_bootstrap  # noqa: E402
from run_detection import load_manifest  # noqa: E402

STOP = set("""a an the of in on at to and or with without is are was were be been being this that these those it its
there their they them image picture photo view aerial overhead top from above below left right center middle side
some several many multiple few various other different large small big tiny long short wide narrow high low
appears seems shows shown visible can see seen into onto by for as near next behind front back up down over under
between around also which while where who whom whose what when how very more most less such one two three four five
area areas part parts parts piece pieces lot lots number each all both any every rest""".split())


def phrases(caption: str, max_n: int = 3) -> list[str]:
    toks = re.findall(r"[a-z]+", caption.lower())
    out, run = set(), []
    for t in toks + ["."]:
        if t in STOP or t == ".":
            for n in range(1, max_n + 1):
                for i in range(len(run) - n + 1):
                    out.add(" ".join(run[i:i + n]))
            run = []
        else:
            run.append(t)
    return sorted(out)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import open_clip
    import torch

    sel_dir = REPO_ROOT / "outputs" / "selection" / "dev"
    cap = json.loads((sel_dir / "captions_florence2_base.json").read_text(encoding="utf-8"))
    vocab = list(csv.DictReader((REPO_ROOT / "data/manifests/lae80c_vocabulary.csv").open(encoding="utf-8")))
    cat_ids = [int(v["category_id"]) for v in vocab]
    names = [prompt_name(v["raw_name"], "light") for v in vocab]
    rows = {int(r["image_id"]): r for r in load_manifest() if r["split"] == "dev"}
    ids = sorted(rows)

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(get_store() / "checkpoints/openai-clip-vit-b-32/open_clip_pytorch_model.bin"))
    tok = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    def enc(texts):
        with torch.no_grad():
            e = model.encode_text(tok(texts)).float()
        return (e / e.norm(dim=-1, keepdim=True)).numpy()

    E_cls = enc(names)
    S = np.full((len(ids), len(cat_ids)), -1.0)
    n_phr = []
    for i, iid in enumerate(ids):
        ph = phrases(cap["captions"].get(str(iid), ""))
        n_phr.append(len(ph))
        if ph:
            S[i] = (enc(ph) @ E_cls.T).max(0)
    np.savez_compressed(sel_dir / "caption_clip.npz", image_ids=np.array(ids), cat_ids=np.array(cat_ids), scores=S,
                        image_ms=np.array([cap["ms"].get(str(i), np.nan) for i in ids], dtype=float),
                        meta=json.dumps({"selector": "caption_clip", "captions": "captions_florence2_base.json",
                                         "captions_meta": cap["meta"], "text_encoder": "openai CLIP ViT-B/32",
                                         "phrases": "1-3-grams of non-stopword tokens", "note": "VocAda-style, not a reproduction"}))

    # evaluation protocol as in M6 (D32)
    cidx = {c: j for j, c in enumerate(cat_ids)}
    G = np.zeros(S.shape, bool)
    for i, iid in enumerate(ids):
        for c in rows[iid]["classes"].split():
            G[i, cidx[int(c)]] = True
    eval_ids = set(json.loads((REPO_ROOT / "outputs/metrics/e0_headroom-e8585910cbfb/run_config.json").read_text())["image_ids"])
    ev = np.array([i in eval_ids for i in ids])
    ca = ~ev

    def setmetrics(M, idx):
        g = G[idx]
        rec = (M & g).sum(1) / g.sum(1)
        present = g.any(0)
        macro = float(np.mean([(M[:, j] & g[:, j]).sum() / g[:, j].sum() for j in np.where(present)[0]]))
        return {"mean_size": float(M.sum(1).mean()), "vocab_recall": float(rec.mean()), "macro_pair_recall": macro, "_rec": rec}

    def topk_mask(Sc, k):
        return np.stack([topk(s, k) for s in Sc])

    rc = np.load(sel_dir / "remoteclip_b32.npz")
    assert list(rc["image_ids"]) == ids
    out = {"n_eval": int(ev.sum()), "n_calibration": int(ca.sum()), "mean_phrases_per_caption": float(np.mean(n_phr)),
           "captions_empty": int(sum(1 for i in ids if not cap["captions"].get(str(i)))), "budgets": {}}
    ev_idx, ca_idx = np.where(ev)[0], np.where(ca)[0]
    for k in (3, 8, 20):
        cap_k = setmetrics(topk_mask(S[ev_idx], k), ev_idx)
        rc_k = setmetrics(topk_mask(rc["scores"][ev_idx], k), ev_idx)
        out["budgets"][str(k)] = {"caption_topk": {x: v for x, v in cap_k.items() if x != "_rec"},
                                  "remoteclip_topk": {x: v for x, v in rc_k.items() if x != "_rec"},
                                  "caption_minus_remoteclip": paired_mean_bootstrap(rc_k["_rec"], cap_k["_rec"])}
    for m in (3, 8):
        t = calibrate(lambda tt: np.mean([threshold(s, tt).sum() for s in S[ca_idx]]), m, -1.0, 1.0, increasing=False)
        mask = np.stack([threshold(s, t) for s in S])
        r = setmetrics(mask[ev_idx], ev_idx)
        out[f"caption_threshold_m{m}"] = {"t": t, **{x: v for x, v in r.items() if x != "_rec"}}
        np.savez_compressed(sel_dir / f"e5_caption_thr_m{m}.npz", image_ids=np.array(ids), cat_ids=np.array(cat_ids),
                            selected=mask, meta=json.dumps({"selector": f"e5_caption_thr_m{m}", "param": t, "protocol": "D32",
                                                            "note": "VocAda-style caption selector, threshold calibrated to mean size"}))
    # TagCLIP-style crop-level analogue from M5 views (OpenAI CLIP, local-only, max over all 29 crops)
    v = np.load(sel_dir / "views_clip_openai_b32.npz")
    assert list(v["image_ids"]) == ids
    T = np.stack([combine_views(v["global_logits"][i], v["crop_logits"][i], v["crop_grid_ids"], 0.0, "max", (2, 3, 4))
                  for i in range(len(ids))])
    np.savez_compressed(sel_dir / "tagclip_style.npz", image_ids=np.array(ids), cat_ids=np.array(cat_ids), scores=T,
                        image_ms=v["image_ms"], meta=json.dumps({"selector": "tagclip_style", "variant": "a0_max_g234",
                                                                  "note": "crop-level analogue of TagCLIP's local-to-global idea; not a reproduction"}))
    for k in (3, 8, 20):
        tk = setmetrics(topk_mask(T[ev_idx], k), ev_idx)
        out["budgets"][str(k)]["tagclip_style_topk"] = {x: val for x, val in tk.items() if x != "_rec"}

    mdir = REPO_ROOT / "outputs/metrics/e5_selection_dev"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "e5_metrics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    L = ["# E5 caption-based (VocAda-style) and TagCLIP-style selectors (generated by `scripts/e5_caption_selector.py`)", "",
         f"Evaluation: the {out['n_eval']} held-out dev images. Mean phrases per caption: {out['mean_phrases_per_caption']:.1f}; "
         f"empty captions: {out['captions_empty']}. Both are labelled baselines, not reproductions.", "",
         "| budget k | selector | vocab recall | macro pair recall |", "|---|---|---|---|"]
    for k, b in out["budgets"].items():
        for name in ("remoteclip_topk", "caption_topk", "tagclip_style_topk"):
            L.append(f"| {k} | {name} | {b[name]['vocab_recall']:.3f} | {b[name]['macro_pair_recall']:.3f} |")
        d = b["caption_minus_remoteclip"]
        L.append(f"| {k} | caption − RemoteCLIP (recall, 95% CI) | {d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]" + ("*" if d["excludes_zero"] else "") + " | |")
    for m in (3, 8):
        r = out[f"caption_threshold_m{m}"]
        L.append(f"| thr m≈{m} | caption threshold (t={r['t']:.3f}) | {r['vocab_recall']:.3f} (size {r['mean_size']:.2f}) | {r['macro_pair_recall']:.3f} |")
    (REPO_ROOT / "reports/generated/E5_selection_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
