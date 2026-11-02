"""M13: one-command check that the committed artifacts are consistent and every table regenerates unchanged.

Usage:
    python scripts/verify_artifacts.py            # dev laptop is enough: needs the repo + <store>/datasets (dev GT)
Checks (no detector, no GPU, no test labels needed):
  1. Frozen split: sha256 of the dev/test ID lists equals D18; no overlap; 730 + 2,862 = 3,592.
  2. Frozen methods: the headline (D34) and both secondaries reproduce the committed masks from the committed
     RemoteCLIP scores on every split that has them (dev, test, nwpu).
  3. Generated tables: re-runs every generator that works from committed files (report_run, report_e0,
     m11_per_source, build_registry, evaluate_selection, m10_vocab_coverage, m11_nwpu_selection) and reports any
     changed file under reports/generated/ or outputs/metrics/. Changed files are restored with git afterwards.
Tables that need gitignored inputs (raw predictions on the GPU machine, M5 views) are listed, not re-generated.
Exit code 0 = everything reproduced.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

SPLIT_SHA = {"dev": "131348ae32b1f97774ced5b5d7aa4e358ae1867b1da3ae9e1570ddd65b2b7f8c",
             "test": "1d6b2c398d3154c9b3ef58572931d0826ceb302f0a12f3f0d0e65380511dbd0e"}
WATCH = ["reports/generated", "outputs/metrics"]
NOT_REGENERATED = {
    "evaluate_run.py (outputs/metrics/<run>/metrics.json)": "needs the run's raw predictions (RTX 5060 only)",
    "m10_failure_analysis.py, m11_nwpu_subgroups.py": "need raw predictions (RTX 5060 only)",
    "m5_global_local.py, m6_retention.py, e9_synonym_selector.py": "need the gitignored M5 views (`run_views.py`, ~minutes on GPU)",
    "e5_caption_selector.py": "needs the Florence-2 captions file (committed) + CLIP text encoder; not re-run here",
    "compare_conditions.py tables (m5_gl_vs_global, m9_tiled_vs_untiled)": "pairs passed on the command line at the time",
}


def git(*a) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO_ROOT).stdout


def check_split() -> list[str]:
    errs, ids = [], {}
    for s, want in SPLIT_SHA.items():
        p = REPO_ROOT / f"data/manifests/lae80c_{s}_ids.txt"
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want:
            errs.append(f"{p.name}: sha256 {got[:12]}… != D18 {want[:12]}…")
        ids[s] = set(p.read_text().split())
    if ids["dev"] & ids["test"]:
        errs.append(f"dev/test overlap: {len(ids['dev'] & ids['test'])} images")
    if (len(ids["dev"]), len(ids["test"])) != (730, 2862):
        errs.append(f"split sizes {len(ids['dev'])}/{len(ids['test'])} != 730/2862")
    return errs


def check_masks() -> list[str]:
    import make_frozen_headline as m
    from aerial_ovd.vocabulary.retention import threshold, top_p

    rules = {"headline_d34": (threshold, m.D34_T), "frozen_threshold_m20": (threshold, 0.004650115966796875),
             "frozen_top_p_m20": (top_p, 0.9464568403151777)}
    dev_names = {"headline_d34": "m6_threshold_m8", "frozen_threshold_m20": "m6_threshold_m20", "frozen_top_p_m20": "m6_top_p_m20"}
    errs, ls = [], m.logit_scale()
    for split in ("dev", "test", "nwpu"):
        d = REPO_ROOT / "outputs/selection" / split
        if not (d / "remoteclip_b32.npz").is_file():
            continue
        z = np.load(d / "remoteclip_b32.npz")
        for name, (rule, p) in rules.items():
            f = d / (f"{dev_names[name]}.npz" if split == "dev" else f"{name}.npz")
            if not f.is_file():
                continue
            c = np.load(f)
            # Dev masks come from the M6 protocol: same rule and parameter, aligned by image id.
            pos = {int(i): j for j, i in enumerate(z["image_ids"])}
            M = m.masks(z["scores"][[pos[int(i)] for i in c["image_ids"]]], ls, rule, p)
            ok = np.array_equal(M, c["selected"])
            print(f"  {split:5s} {f.name:28s} {'reproduced' if ok else 'MISMATCH'} (mean size {M.sum(1).mean():.2f})")
            if not ok:
                errs.append(f"{split}/{f.name}: {int((M != c['selected']).any(1).sum())} images differ")
    return errs


def regenerate() -> list[list[str]]:
    py = sys.executable
    cmds = []
    for mdir in sorted((REPO_ROOT / "outputs/metrics").iterdir()):
        if (mdir / "metrics.json").is_file() and (mdir / "run_config.json").is_file():
            cmds.append([py, "scripts/report_run.py", mdir.relative_to(REPO_ROOT).as_posix()])  # path is echoed in the header
    cmds.append([py, "scripts/report_e0.py", "outputs/metrics/e0_headroom-e8585910cbfb"])
    for run in ("e5_caption_tagclip_det-69e357c3b1f4", "m4_selectors_det-16d38bd056f8", "final_test-75ab843910a3"):
        cmds.append([py, "scripts/m11_per_source.py", f"outputs/metrics/{run}"])
    cmds += [[py, "scripts/build_registry.py"], [py, "scripts/evaluate_selection.py", "--split", "dev"],
             [py, "scripts/m10_vocab_coverage.py"], [py, "scripts/m11_nwpu_selection.py"]]
    return cmds


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    errs = []
    print("1. frozen split"); e = check_split(); errs += e; print("  ok" if not e else "\n".join("  " + x for x in e))
    print("2. frozen method masks"); errs += check_masks()
    if git("status", "--porcelain", "--", *WATCH).strip():
        print("3. skipped: reports/generated or outputs/metrics has uncommitted changes (commit or stash first)")
        errs.append("dirty generated/metrics tree")
    else:
        print("3. regenerating tables from committed files")
        failed = []
        for c in regenerate():
            r = subprocess.run(c, capture_output=True, text=True, cwd=REPO_ROOT)
            name = " ".join(Path(x).name if i else "python" for i, x in enumerate(c))
            if r.returncode:
                failed.append(name)
                print(f"  FAILED: {name}\n    {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}")
        changed = [l[3:] for l in git("status", "--porcelain", "--", *WATCH).splitlines()]
        for f in changed:
            print(f"  CHANGED: {f}")
        print(f"  {len(regenerate())} generators run, {len(failed)} failed, {len(changed)} files changed")
        errs += [f"generator failed: {f}" for f in failed] + [f"changed: {f}" for f in changed]
        if changed:  # restore the committed versions (tracked files) and remove new untracked outputs
            git("checkout", "--", *WATCH)
            git("clean", "-fdq", "--", *WATCH)
    print("Not regenerated here (inputs not in git):")
    for k, v in NOT_REGENERATED.items():
        print(f"  - {k}: {v}")
    print("ALL REPRODUCED" if not errs else f"{len(errs)} PROBLEM(S)")
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
