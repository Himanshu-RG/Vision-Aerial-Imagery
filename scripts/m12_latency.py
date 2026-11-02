"""M12: end-to-end latency on an idle GPU (D49). Selector and detector timed in ONE process, per image.

Usage (RTX 5060):
    python scripts/m12_latency.py                      # 200-image dev sample (E0 sample), 3 passes
    python scripts/m12_latency.py --n-images 3 --repeats 1 --device cpu --allow-busy   # smoke test only
Writes outputs/metrics/m12_latency/<host>.json (small, committed) and reports/generated/M12_latency.md.

Pipelines (interleaved per image, order rotated per image so no pipeline always runs first):
    full      detector with all 80 classes (no selector)
    headline  RemoteCLIP global selector -> p_g = softmax(logit_scale * cosine) >= t (D34) -> detector
    rc_top8   RemoteCLIP global selector -> fixed top-8 -> detector
Latency = selector (image preprocess + encode + similarity + rule; text embeddings cached once per vocabulary)
        + detector (preprocess + forward + postprocess), CUDA-synchronised. Disk I/O and model loading excluded (D22).
Idle check (D9, D49a): GPU utilisation <= 10% in all of 10 samples taken 1 s apart before the models load; otherwise
the script stops unless --allow-busy is given (then the result is labelled 'not idle'). The processes nvidia-smi lists
and whether the screen is drawn by this GPU are recorded, not used as the criterion: on Windows every program on a
screen attached to the GPU is listed. Dev images only: the test split is used up.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aerial_ovd.data.splits import image_dir  # noqa: E402
from aerial_ovd.utils.paths import get_store  # noqa: E402
from aerial_ovd.vocabulary.names import prompt_name  # noqa: E402
from aerial_ovd.vocabulary.retention import threshold, topk  # noqa: E402
from aerial_ovd.vocabulary.selectors import CLIPSelector, softmax  # noqa: E402
from make_frozen_headline import D34_T, logit_scale  # noqa: E402
from run_detection import gpu_snapshot, load_manifest, sample_images  # noqa: E402

PIPELINES = ("full", "headline", "rc_top8")
MAN = REPO_ROOT / "data" / "manifests"


def smi(query: str) -> list[str]:
    out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout.strip()
    return [x.strip() for x in out.splitlines()[0].split(",")] if out else []


def idle_samples(n: int = 10, interval_s: float = 1.0) -> dict:
    """GPU utilisation and memory sampled n times before the models load (D49a)."""
    util, mem = [], []
    for _ in range(n):
        u, m = smi("utilization.gpu,memory.used")
        util.append(float(u))
        mem.append(float(m))
        time.sleep(interval_s)
    try:
        display_active, display_mode = smi("display_active,display_mode")
    except Exception:  # noqa: BLE001
        display_active = display_mode = "unknown"
    return {"util_pct": util, "mem_used_mb": mem, "max_util_pct": max(util), "display_active": display_active,
            "display_mode": display_mode}


def compute_apps() -> list[dict]:
    """Processes nvidia-smi lists on the GPU. Recorded only: on Windows (WDDM) this includes every program that
    draws on a screen attached to the GPU (explorer.exe, editors, ...), so it is not an idle criterion (D49a)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return [{"error": repr(e)}]
    return [dict(zip(("pid", "name", "used_mb"), [x.strip() for x in line.split(",")])) for line in out.splitlines() if line.strip()]


def median_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Median and 95% percentile bootstrap CI over images."""
    rng = np.random.default_rng(seed)
    b = np.median(x[rng.integers(0, len(x), (n_boot, len(x)))], axis=1)
    return float(np.median(x)), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3, help="passes over the sample; every pass is recorded")
    ap.add_argument("--warmup", type=int, default=5, help="unrecorded warm-up images per pipeline")
    ap.add_argument("--allow-busy", action="store_true")
    args = ap.parse_args()

    import torch
    from PIL import Image

    from aerial_ovd.detector.gdino import GroundingDINO

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available -- wrong venv?", file=sys.stderr)
        return 2
    on_gpu = args.device.startswith("cuda")
    gpu0, apps0 = gpu_snapshot(torch), compute_apps() if on_gpu else []
    others = [a for a in apps0 if str(a.get("pid")) != str(os.getpid())]
    samples = idle_samples() if on_gpu else {}
    # Idle (D49a) = GPU utilisation <= 10% in every one of 10 samples over ~10 s before the models load.
    idle = on_gpu and samples["max_util_pct"] <= 10
    if not idle and not args.allow_busy:
        print(f"GPU not idle: utilisation samples {samples.get('util_pct')} % (all must be <= 10). Close games, "
              "Overwolf, browsers and video, plug in, Best performance, and retry. --allow-busy only for smoke tests.",
              file=sys.stderr)
        return 3
    if on_gpu:
        print(f"idle check passed: max utilisation {samples['max_util_pct']:.0f}%, memory used "
              f"{max(samples['mem_used_mb']):.0f} MB, display on this GPU: {samples['display_active']}", flush=True)

    store = get_store()
    rows = load_manifest()
    by_id = {int(r["image_id"]): r for r in rows}
    ids = sample_images(rows, "dev", args.n_images, 2026)  # the E0/M4-M7 sample (D22) when n = 200
    vocab = list(csv.DictReader((MAN / "lae80c_vocabulary.csv").open(encoding="utf-8")))
    names = [prompt_name(v["raw_name"], "light") for v in vocab]
    cat_ids = [int(v["category_id"]) for v in vocab]
    assert cat_ids == sorted(cat_ids)
    ls = logit_scale()
    committed = np.load(REPO_ROOT / "outputs/selection/dev/m6_threshold_m8.npz")
    committed_mask = {int(i): committed["selected"][j] for j, i in enumerate(committed["image_ids"])}

    ck = store / "checkpoints"
    t = time.perf_counter()
    det = GroundingDINO(ck / "grounding-dino-tiny", device=args.device, precision="fp32", num_select=900)
    sel = CLIPSelector("remoteclip_b32", "ViT-B-32", ck / "remoteclip-vit-b-32" / "RemoteCLIP-ViT-B-32.pt", names,
                       device=args.device, state_dict_format=True)
    load_s = time.perf_counter() - t

    def run(pipe: str, image) -> dict:
        rec = {}
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        if pipe == "full":
            vocab_idx, rec["selector_ms"] = np.arange(len(names)), 0.0
        else:
            t0 = time.perf_counter()
            s = sel.scores(image)  # returns a CPU array, so the GPU work is finished
            m = threshold(softmax(ls * s), D34_T) if pipe == "headline" else topk(s, 8)
            vocab_idx = np.flatnonzero(m)
            rec["selector_ms"] = (time.perf_counter() - t0) * 1e3
            rec["mask"] = m
        d = det.detect(image, [names[k] for k in vocab_idx])
        rec.update({f"det_{k}_ms": v for k, v in d.timings_ms.items()})
        rec["detector_ms"] = sum(d.timings_ms.values())
        rec["end_to_end_ms"] = rec["selector_ms"] + rec["detector_ms"]
        rec["n_classes"] = int(len(vocab_idx))
        if args.device.startswith("cuda"):
            rec["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        return rec

    img_dir = image_dir(store, "dev")

    def load(iid):
        with Image.open(img_dir / by_id[iid]["file_name"]) as im:
            return im.convert("RGB")

    for iid in ids[: args.warmup]:
        image = load(iid)
        for p in PIPELINES:
            run(p, image)

    recs, mismatch = [], 0
    for rep in range(args.repeats):
        for n, iid in enumerate(ids):
            image = load(iid)
            order = PIPELINES[n % 3:] + PIPELINES[: n % 3]
            for p in order:
                r = run(p, image)
                m = r.pop("mask", None)
                if p == "headline" and rep == 0 and not np.array_equal(m, committed_mask[iid]):
                    mismatch += 1
                recs.append({"rep": rep, "image_id": iid, "pipeline": p, **{k: (round(v, 3) if isinstance(v, float) else v)
                                                                             for k, v in r.items()}})
        print(f"pass {rep + 1}/{args.repeats} done", flush=True)
    gpu1 = gpu_snapshot(torch)

    # Summary: per image, median over passes; then median over images with bootstrap CI.
    def per_image(p, key):
        return np.array([np.median([r[key] for r in recs if r["pipeline"] == p and r["image_id"] == i]) for i in ids])

    summ = {}
    full_e2e = per_image("full", "end_to_end_ms")
    for p in PIPELINES:
        e2e = per_image(p, "end_to_end_ms")
        summ[p] = {"end_to_end_ms": median_ci(e2e), "selector_ms": median_ci(per_image(p, "selector_ms")),
                   "detector_ms": median_ci(per_image(p, "detector_ms")),
                   "forward_ms": median_ci(per_image(p, "det_forward_ms")),
                   "p90_end_to_end_ms": float(np.quantile(e2e, 0.9)),
                   "mean_classes": float(np.mean(per_image(p, "n_classes"))),
                   "peak_mem_mb_max": max((r.get("peak_mem_mb", 0) for r in recs if r["pipeline"] == p), default=None),
                   "diff_vs_full_ms": median_ci(e2e - full_e2e),
                   "speedup_vs_full": float(np.median(full_e2e) / np.median(e2e))}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs"], capture_output=True,
                                text=True, cwd=REPO_ROOT).stdout.strip())
    out = {"protocol": "D49", "host": socket.gethostname(), "device": args.device, "git_commit": git, "git_dirty_code": dirty,
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
           "idle_at_start": bool(idle), "idle_criterion": "D49a: max of 10 utilisation samples (1 s apart) <= 10%",
           "idle_samples": samples, "gpu_at_start": gpu0, "gpu_processes_at_start": apps0, "gpu_at_end": gpu1,
           "n_images": len(ids), "repeats": args.repeats, "warmup": args.warmup, "model_load_s": round(load_s, 2),
           "text_cache_ms": sel.text_cache_ms, "headline_mask_mismatch_vs_committed": mismatch,
           "summary": summ, "records": recs}
    dst = REPO_ROOT / "outputs/metrics/m12_latency"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f"{socket.gethostname()}.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")

    f = lambda c: f"{c[0]:.1f} [{c[1]:.1f}, {c[2]:.1f}]"  # noqa: E731
    L = [f"# M12 end-to-end latency (generated by `scripts/m12_latency.py`; protocol D49)", "",
         f"Host `{out['host']}` · {out['gpu_name']} · torch {out['torch']} · git `{git[:7]}` (code dirty={dirty}) · "
         f"**GPU idle at start: {'yes' if idle else 'NO, indicative only'}** (max utilisation "
         f"{samples.get('max_util_pct', '–')}% over 10 samples, {gpu0.get('smi_used_mb')} MB used, display on this GPU: "
         f"{samples.get('display_active', '–')}, {len(others)} processes listed by nvidia-smi) · {len(ids)} dev images × {args.repeats} "
         f"passes · headline masks differing from the committed CPU masks: {mismatch}/{len(ids)}", "",
         "Per image: median over passes; then median over images with a 95% bootstrap CI. fp32, top-900 (D21).", "",
         "| pipeline | mean classes | selector ms | detector ms | forward ms | **end-to-end ms** | p90 | Δ vs full ms | speed-up | peak MB |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for p, s in summ.items():
        peak = "–" if not s["peak_mem_mb_max"] else f"{s['peak_mem_mb_max']:.0f}"
        diff = "–" if p == "full" else f(s["diff_vs_full_ms"])
        L.append(f"| {p} | {s['mean_classes']:.1f} | {f(s['selector_ms'])} | {f(s['detector_ms'])} | {f(s['forward_ms'])} | "
                 f"**{f(s['end_to_end_ms'])}** | {s['p90_end_to_end_ms']:.1f} | {diff} | ×{s['speedup_vs_full']:.2f} | {peak} |")
    (REPO_ROOT / "reports/generated/M12_latency.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
