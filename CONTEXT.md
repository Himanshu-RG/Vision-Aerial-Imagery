# CONTEXT.md — Project Ground Truth

> **Single source of truth for this project.** Any agent or person resuming work must read this file first.
> It is updated at the end of every phase (module). If this file conflicts with memory, chat history, or other docs, **this file wins** — unless it is stale, in which case fix it first.
> Last updated: 2026-10-20 · Phase: **M0 + M1 finished on the dev laptop (commit 1256354); RTX 5060 audit pending · next: M2**

---

## 1. Project identity
- **Title:** Adaptive Vocabulary Selection for Robust Open-Vocabulary Object Detection in Aerial Imagery
- **Team:** Group "Transmontane": Amit Adhikari (23BCE2327), Himanshu Ray (23BCE2335), Saurav Kumar Gupta (23BCE2336), Avishek Chowdhury (23BCE2344)
- **Source documents (in repo root, not committed):**
  - `Brief_Project_Description_Adaptive_Vocabulary_Aerial_OVD.pdf`: the official project brief.
  - `01.pdf`: unrelated paper (KnowScene, Knowledge-Based Systems 2026, 3D scene understanding). Not used.
- **Master instructions:** a "Master Research & Implementation Prompt" (14 modules, M0–M14) given by the user on 2026-10-20. Its rules are summarized in §3; the module list is in §7.

## 2. Central research question (fixed — do not change)
Does **image-conditioned, adaptive vocabulary selection** improve the accuracy, recall, robustness, and computational efficiency of a **frozen, pretrained Grounding DINO** on aerial imagery?
The research directions to investigate are (a) global–local category relevance, (b) uncertainty-aware vocabulary retention, and (c) adaptive vocabulary budgets.
**Vocabulary pruning itself is NOT claimed as novel** (VocAda already does it for natural images). No new detector, and no fine-tuning of Grounding DINO in the initial study.

## 3. Non-negotiable rules (from master prompt — apply to every agent)
1. Never fabricate results, metrics, citations, or claims of execution.
2. Make no novelty claim until prior work has been checked; see §5.
3. **No test-set leakage.** Test annotations never influence selection, thresholds, hyperparameters, or decisions. Tune on the dev split only.
4. Comparisons must be fair: same split, checkpoint, classes, preprocessing, evaluation, and hardware.
5. Start with pretrained inference; keep the detector frozen.
6. Build simple baselines before complex methods.
7. No silent failures: log failed images, OOMs, invalid boxes, and empty predictions. Failed images stay in the denominator.
8. Thresholds, budgets, and hyperparameters are configurable and validated on the dev split.
9. Latency includes the selector, crops, preprocessing, detector, and post-processing. Warm and cold timings are reported separately.
10. Record commands, configs, checkpoint SHAs, seeds, versions, and hardware for every run.
11. Report negative results honestly.
12. Work module by module: **Inspect → Design → Implement → Test → Validate → Record → Proceed.** Each module ends with a report covering objective, files, commands, tests, actual results, limitations, failure cases, exit criteria, and next steps.
13. Oracle analyses (using GT categories or boxes) must always be labelled **"oracle diagnostic"** and never presented as methods.

## 4. Compute & environment (dev laptop audited 2026-10-20; see `reports/M1_environment.md`)
| | Dev laptop `LAPTOP-8N8B1VIP` (audited) | Experiment machine (**not audited yet**) |
|---|---|---|
| Role | coding, unit tests, CPU smoke tests on a small subset | **all real experiments + all latency measurements** |
| GPU | **NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB** (driver 581.08; 331 MB in use at audit). Deliberately **unused**: CPU torch only (shared with games, different arch from the experiment GPU) | **RTX 5060, 8 GB** VRAM (Blackwell, sm_120) |
| Disk free | C: (repo, OneDrive) 4.1 GB · D: (store) **5.9 GB after the venv** | ~100 GB free |
| CPU / RAM | i7-11800H (env_audit reports 8 logical CPUs) / 15.7 GB | TBD (run `scripts/env_audit.py`) |
| OS | Windows 11 Home, build 10.0.26300 (Python reports `Windows-10-10.0.26300`) | TBD |
| Python / venv | 3.11.9; venv **`D:\aerial-ovd-store\venv`** (system torch 2.2.2+cpu is not used) | TBD |
| Key versions | torch 2.7.1+cpu, torchvision 0.22.1+cpu, transformers 4.57.3, huggingface_hub 0.36.2, open_clip_torch 3.3.0, pycocotools 2.0.11, numpy 2.4.6, pydantic 2.14.1 | TBD (must match, except torch cu128) |
| Records | `outputs/env/LAPTOP-8N8B1VIP.json`, `requirements-lock-LAPTOP-8N8B1VIP.txt` (50 packages) | pending: `outputs/env/<host>.json`, `requirements-lock-<host>.txt` |
| Tests | **16/16 pass** (`pytest`, 7.38 s) | pending |

- **Sync:** via GitHub, remote `origin` = https://github.com/Himanshu-RG/Vision-Aerial-Imagery.git (branch `main`, first pushed 2026-10-20). Commits use the machine's global git identity (`Himanshu-RG`).
- **Pinned stack (both machines):** torch 2.7.1 / torchvision 0.22.1. Use the CPU wheels on the laptop and the **cu128** wheels on the RTX 5060 (Blackwell needs torch ≥ 2.7 + CUDA 12.8). transformers 4.57.3. `requirements.txt` (CPU) and `requirements-gpu.txt` (cu128; same non-torch pins). Exact locks go in `requirements-lock-<host>.txt`.
- **Checkpoints** (`configs/checkpoints.yaml`; dry-run 2026-10-20 resolved these commits; **not downloaded yet**; `configs/checkpoints.lock.json` with sha256 is written on the first real download):
  - `IDEA-Research/grounding-dino-tiny@a2bb814dd3…` (658.3 MB)
  - `chendelong/RemoteCLIP@bf1d8a3ccf…` (577.2 MB)
  - `timm/vit_base_patch32_clip_224.openai@a6f597a30f…` (577.2 MB)
  - Total 1.77 GB, which fits on D: (5.9 GB free) with a ~1 GB margin. Download is deferred to M2/M3.
- **Large-file store:** `D:\aerial-ovd-store` on the laptop, which is outside OneDrive so large files don't sync. Its location is set by the `AOVD_STORE` env var or by `configs/paths.local.yaml` (gitignored). Set `HF_HOME=<store>/hf_cache`.
- **Laptop data policy:** keep only a stratified subset (~1k dev images) of LAE-80C locally. Delete the zip after extraction.

## 5. Literature state (Module 0) — verified facts only
Full matrix with 13 works and verification levels ([FT] full text via LLM-assisted query, [AB] abstract/landing page, [META] metadata only): `reports/M0_literature_matrix.md`. Re-check exact numbers before quoting them in the paper.

| Work | Status / key facts | Relevance |
|---|---|---|
| **Grounding DINO** (Liu et al., ECCV 2024; arXiv 2303.05499; github IDEA-Research/GroundingDINO) [META+repo] | Detector used here. BERT text encoder, `max_text_len` = **256** in the grounding-dino-tiny config. HF checkpoint `IDEA-Research/grounding-dino-tiny`. | Primary detector |
| **VocAda** (Liu, Hayes, Mancini, Ricci, Volpi, Csurka; ICIP 2025, pp. 540–545; arXiv 2506.00333; github OatmealLiu/VocAda) [FT] | Training-free pipeline: LLaVA-Next-7B caption → spaCy noun phrases → class selection by CLIP-text top-k (k=1 default) or by LLM. Detectors: Detic, VLDet, CoDet (**not GDINO**). Data: COCO OVE, Objects365 CDTE (**not aerial**). Gains up to +3.3 AP (COCO) and +2.2 AP (O365). Overhead 5.6 s/img with 17 GB (CLIP variant) or 10.7 s/img with 31 GB (LLM variant) on a V100. Retains 94.8% (CLIP) and 86.1% (LLM) of relevant classes. | Closest method; baseline E5 ("VocAda-style", not a reproduction) |
| **Tsourveloudis, "Do Open-Vocabulary Detectors Transfer to Aerial Imagery? A Comparative Evaluation"** (arXiv 2601.22164, Jan 2026) [FT] | Zero-shot evaluation on LAE-80C of GDINO, OWLv2, YOLO-World, YOLOE, and LLMDet. **Oracle per-image vocabulary (~3.2 classes) gives ~15× higher F1 than the 80-class prompt** (0.5% → 7.4%). Prompt engineering ("Aerial view of", synonyms) does not help. Metrics are P/R/F1 with IoA 0.7, not COCO mAP. **No deployable selector.** It **explicitly calls for "intelligent vocabulary filtering" as future work.** It does not discuss the 256-token limit. | Closest study; motivates headroom; our E0 replicates its direction with mAP |
| **LAE-DINO / LAE-1M** (Pan et al., AAAI 2025, 39(6):6281–6289; arXiv 2408.09110; github jaychempan/LAE-DINO; HF `jaychempan/LAE-1M`, MIT) [AB+repo+HF] | Dynamic Vocabulary Construction is a **training-time** mechanism. LAE-80C benchmark: 80 classes, COCO JSON, images.zip 1.97 GB, built from DOTA-v2 / DIOR / FAIR1M / xView val splits. Image count is **unresolved** (3,592 per a citing paper vs. ~20k (val) per the HF README; resolve in M2). | Primary dataset source |
| **TagCLIP** (Lin et al., AAAI 2024, 38:3513–3521; arXiv 2312.12828; github linyq2117/TagCLIP) [AB+repo] | Training-free **local-to-global** CLIP multi-label classification for natural images (VOC2007 92.8 mAP, COCO2014 68.7 mAP). Tagging only, no downstream detector. | Direct overlap with the global–local idea; must be a baseline |
| **CastDet** (Y. Li et al., ECCV 2024; arXiv 2311.11646; github lizzy8587/CastDet) [AB] | **Training-based** CLIP-activated student–teacher self-training with **RemoteCLIP as an extra teacher**, plus a dynamic label queue. **46.5% mAP on VisDroneZSD novel classes.** No vocabulary selection. | Aerial OVD context; supports RemoteCLIP for aerial semantics |
| **OpenRSD** (Z. Huang et al., ICCV 2025, pp. 8384–8394; arXiv 2503.06146; github floatingstarZ/OpenRSD, release not verified) [FT] | **Training-based** open-prompt RS detector: SkyCLIP text encoder + DINOv2 image prompts; three-stage training including ORSD+ (**470k images, 200 classes**). **20.8 FPS on a 2080Ti.** No selection at inference. | Aerial OVD context; not comparable as a selector |
| **Zhao et al., CVPR 2024, SAS** (Scene-adaptive & Region-aware Multi-modal Prompt; pp. 16741–16750) [FT] | **Learned** scene and region prompts on a Faster R-CNN R50 (training on base classes); OV-COCO / OV-LVIS (+ VOC transfer). It **modifies text embeddings and does not remove classes.** Natural images. | Scene-adaptive prompting (training-based); not a selector |
| **RT-OVAD** (Wei et al., arXiv 2408.12246v3, Jul 2025; venue not verified; github GT-Wei/RT-OVAD) [FT] | **Training-based** real-time aerial OVD (RT-DETR + image-text alignment) trained on LAE-1M. ZSD/GZSD splits on xView, DIOR, DOTA. **Its GDINO baselines are trained, so they are not comparable** to our frozen zero-shot GDINO (exact setting not fully verified). | Aerial OVD context |
| **FACTOR** (Zhao et al., ICML 2026; arXiv 2605.03294) [FT] | Training-free counterfactual TTA that corrects GDINO logits under corruptions. Does not select vocabulary. Not aerial. | Adjacent (OVOD test-time adaptation) |
| **Li & Tang, CMC 2026** ("Adapting GroundingDINO for RS detection… concise category-aware prompts") [FT] | Fine-tunes GDINO on DOTA/DIOR. Finds that concise 2–3-token prompts work best. No per-image selection. | Prompt-naming evidence (M2 naming policy) |
| **Vashisht, NAECON 2026** (zero-shot SAM 3 on drone imagery) [META, citation context] | Cites 2601.22164 and lists prompt-vocabulary reduction as a future direction. | Confirms the gap is recognised but open |
| **PISA** (He, Yang, Wang; arXiv 2608.14142, Aug 2026) [META] | Test-time OVOD via pseudo-individual source-domain feature adaptation. Content not read. | **Pending:** check before submission |

**Citation sweep (Semantic Scholar, 2026-10-20):** VocAda (2506.00333) has 5 citing papers, and none does deployable vocabulary selection for aerial imagery. 2601.22164 has 1 citing paper (Vashisht, NAECON 2026). **No work was found that performs deployable, inference-time vocabulary selection for a frozen OV detector on aerial imagery, so the gap holds as of 2026-10-20.** Repeat this sweep before any submission.

**Pending pre-submission checks** (matrix §4): full-text read of CastDet's protocol (which classes are novel per dataset); RT-OVAD venue and exact GDINO baseline setting; PISA content; conformal multi-label prediction references for M6; the exact LAE-80C image count (M2).

**Gap statement** (`reports/M0_gap_hypotheses_novelty.md` §1): Prior work has shown the oracle headroom of vocabulary reduction for aerial OVD. It has not tested whether a *deployable, inference-time* selector recovers that headroom for a frozen Grounding DINO, measured with COCO metrics, small- and rare-object breakdowns, the 256-token limit, and end-to-end latency.

**Novelty risk:**
- Vocabulary pruning itself: HIGH risk. Not claimed.
- Global CLIP selection: HIGH risk. Used only as baseline D.
- Global–local evidence: MEDIUM-HIGH risk (TagCLIP). Claim only an aerial-specific, detector-coupled design, with a TagCLIP-style baseline.
- Uncertainty-aware retention: MEDIUM risk (conformal multi-label prediction exists). Candidate rule: calibrate the selector on dev to a target vocabulary recall (coverage), and report the achieved coverage on test.
- Adaptive budget: HIGH risk as a standalone contribution. Framed as a consequence of the coverage rule.
- Empirical study: LOW-MEDIUM risk. This is the main defensible contribution together with the selector.

**Realistic venue:** an EarthVision-type workshop, IGARSS, GRSL, or JSTARS, unless effects are large and robust.

## 6. Hypotheses (falsifiable; full table with falsification criteria in `reports/M0_gap_hypotheses_novelty.md` §2)
- **H0 (confound):** the 80 class names exceed GDINO's 256-token limit, so the full vocabulary needs chunked prompts. Part of the prior study's collapse may be truncation.
  - **Result (2026-10-20): FALSIFIED for raw names.** `python scripts/check_prompt_length.py --classes configs/lae80c_classes_raw.txt` (tokenizer grounding-dino-tiny@a2bb814dd3) gives a prompt of **251 tokens** including [CLS]/[SEP]. The format is `"name1. name2. … name80."`, lower-cased. That is ≤ 256 (`max_text_len`), so there is **no truncation**. Tokens per class: min 2, max 8 (`Front loader/Bulldozer`), mean 3.11.
  - **Caveat:** the margin is only **5 tokens**. Any prefix ("aerial view of"), synonym expansion, or longer naming policy will exceed 256. **Re-check whenever the naming policy changes** (M2). Whether the prior study's prompt variants were truncated is unverified.
- **H1:** oracle vocabulary > full vocabulary in mAP@50 on dev, with a bootstrap CI excluding 0 (the headroom).
- **H2:** some deployable global selector at a matched budget beats the full vocabulary.
- **H3:** local evidence raises vocabulary recall for small-object classes at a matched mean vocabulary size.
- **H4:** uncertainty-aware retention raises detection recall at a matched mean vocabulary size.
- **H5:** an adaptive budget Pareto-dominates fixed-k.
- **H6:** end-to-end latency, including the selector, is below that of the full vocabulary. For raw names that is a single prompt, not chunked (see H0).
- **Gate:** M6/M7 proceed only if M4/M5 show a selector beating the full vocabulary on dev (H2 holds). If H1 fails, pivot to an analysis paper. If H1 holds but H2 fails, report the negative result and analyse selection errors before M5.

## 7. Module roadmap & status
| Module | Description | Status |
|---|---|---|
| M0 | Literature audit & novelty | ✅ exit criteria met (`reports/M0_gap_hypotheses_novelty.md` §5). Pre-submission re-checks remain open (§5) |
| M1 | Repo & compute audit | 🟡 done on the dev laptop (exit criteria ✅ there). Pending: RTX 5060 audit + GPU lock file |
| M2 | Dataset selection & validation, frozen protocol | ⬜ **next** (see `reports/M1_environment.md` "Next") |
| M3 | Grounding DINO baseline | ⬜ |
| M4 | Master vocab + simple baselines (A full, B fixed-reduced, C random, D global CLIP, E VocAda-style) | ⬜ |
| M5 | Global–local relevance | ⬜ |
| M6 | Uncertainty-aware retention | ⬜ (gated) |
| M7 | Adaptive budget | ⬜ (gated) |
| M8 | Unified experiment runner (E1–E8) | ⬜ (grows with M3–M4) |
| M9 | Metrics + validation on known examples | ⬜ (must pass before M3 numbers are trusted) |
| M10 | Failure analysis | ⬜ |
| M11 | Cross-dataset generalization | ⬜ |
| M12 | Efficiency / resume / profiling | ⬜ (cross-cutting) |
| M13 | Reproducibility & tests | ⬜ (cross-cutting; 16 utility tests exist) |
| M14 | Paper / publication readiness | ⬜ |

Dependency chain: M0 → M2. M1 → M2 → M3 → M4 → M5 → M6 → M7 → M10/M11 → M14.

**M2 to-do** (from `reports/M1_environment.md`):
1. Download the LAE-80C JSON (45 MB) and resolve the image count (3,592 vs. ~20k).
2. Build the manifest.
3. Validate boxes and categories.
4. Freeze the dev/test split (D2).
5. Decide the class-naming policy, then re-run the H0 token check.
6. Decide the VisDrone mapping (D3).

## 8. Frozen / chosen decisions
| ID | Decision | Reason | Date |
|---|---|---|---|
| D1 | Primary dataset: **LAE-80C** | 80 classes (selection matters), COCO format, MIT license, used by the closest prior study | 2026-10-20 (user) |
| D2 | Split LAE-80C (one official set) once, seeded and stratified by source dataset, into **dev 20% / test 80%**. Commit ID lists before any detector run | No test leakage | 2026-10-20 (default) |
| D3 | Generalization: per-source-domain breakdown + VisDrone-DET val as external set (mapping decided in M2) | DIOR images overlap LAE-80C | 2026-10-20 (default) |
| D4 | Detector: HF `IDEA-Research/grounding-dino-tiny`, frozen, pinned commit SHA | 8 GB GPU; avoids compiling CUDA ops on Windows | 2026-10-20 |
| D5 | Token→class mapping via positive-map (max over class token span), not HF phrase strings | Correct per-class scores | 2026-10-20 |
| D6 | Full vocabulary = chunked prompts if > 256 tokens; chunking rule documented | H0 | 2026-10-20 |
| D7 | Selector encoders: OpenAI CLIP ViT-B/32 vs RemoteCLIP ViT-B/32. Caveat: RemoteCLIP saw captions derived from DOTA/DIOR **train** annotations | Domain gap | 2026-10-20 |
| D8 | VocAda-style baseline uses a small captioner (e.g. Florence-2-base); labelled "VocAda-style, not reproduction" | LLaVA-7B does not fit in 8 GB | 2026-10-20 |
| D9 | Latency measured only on the RTX 5060, with the GPU otherwise idle | Fair timing | 2026-10-20 |
| D10 | Refines D6: chunk **only if** a prompt exceeds 256 tokens. Raw LAE-80C names fit in a single prompt (251 tokens), so the full vocabulary (A/E0) is **one prompt**. Re-run `scripts/check_prompt_length.py` after any naming change; chunking applies only to variants that exceed the limit | H0 falsified for raw names (§6) | 2026-10-20 |
| D11 | Refines D4: checkpoints pinned to the commits resolved on 2026-10-20 (GDINO-T `a2bb814dd3…`, RemoteCLIP `bf1d8a3ccf…`, timm OpenAI CLIP `a6f597a30f…`). The sha256 lock (`configs/checkpoints.lock.json`) is written on the first download and used thereafter | Reproducibility (rule 10) | 2026-10-20 |
| D12 | The dev laptop uses CPU-only torch. Its 4 GB RTX 3050 Ti is not used for any run | Shared with games; different architecture; avoids mixing hardware (rule 4, D9) | 2026-10-20 |

## 9. Repository layout
```
vision/                      (git repo root, branch main; remote origin = github.com/Himanshu-RG/Vision-Aerial-Imagery)
  CONTEXT.md                 ← this file (ground truth)
  README.md, pyproject.toml, requirements.txt (CPU), requirements-gpu.txt (cu128), requirements-lock-<host>.txt
  .claude/agents/context-keeper.md   agent that updates this file after each phase
  configs/                   paths.example.yaml (committed), paths.local.yaml (ignored), checkpoints.yaml,
                             checkpoints.lock.json (after first download), lae80c_classes_raw.txt (80 raw names)
  data/manifests/            split ID lists, dataset manifests (committed, small)
  src/aerial_ovd/{data,detector,vocabulary,evaluation,experiments,utils}/   (utils: seed, logging, config, run_id, paths)
  scripts/                   env_audit.py, fetch_checkpoints.py, check_prompt_length.py
  tests/                     16 unit tests
  outputs/{logs,predictions,metrics,figures,env}/   (predictions ignored; env + metrics committed)
  reports/                   per-module reports (M0_literature_matrix.md, M0_gap_hypotheses_novelty.md, M1_environment.md, ...)
```

## 10. First experiment queued (after M2/M3 setup): E0 headroom check
~200 stratified dev images on the RTX 5060, GDINO-T fp16. Conditions:
- (a) full 80-class vocabulary as a **single prompt**. For raw names it is 251 tokens, so no chunking is needed (D10). If M2 changes the naming, re-check, and chunk only if the prompt exceeds 256.
- (b) **oracle** per-image GT vocabulary (diagnostic).
- (c) random vocabulary size-matched to the oracle.

Metrics: mAP@50, mAP@50:95, recall@100, latency, peak VRAM, failures.
- **Success:** 0 unexplained failures; pycocotools passes the known-answer test (GT as predictions gives AP = 1.0); oracle > full with a CI excluding 0.
- **Rethink if:** oracle ≈ full (no headroom), or oracle mAP@50 ≈ 0 (detector too weak, so address tiling first).

## 11. Risks / open issues
1. GDINO-T may be near the floor on aerial images (the prior study saw ~4% recall even with an oracle vocabulary). Tiling may be needed, applied identically to all methods.
2. 256-token limit: resolved for raw names (251/256), but the **5-token margin** means any naming, prefix, or synonym change can silently truncate. Always re-run the token check (D10).
3. **LAE-80C image count is unresolved** (3,592 per a citing paper vs. ~20k val per the HF README). It affects the split sizes (D2), the laptop subset, and comparability with 2601.22164. Resolve from the JSON in M2.
4. **LAE-80C class names have naming problems:**
   - concatenated names (`groundtrackfield`, `storagetank`, …);
   - mixed case;
   - hierarchy overlap (`vehicle` vs `Small Car`; `airplane` vs `Fixed-wing Aircraft`).

   These can hurt text grounding and create ambiguous "wrong-class" detections. The naming policy is an M2 decision, and it must be applied identically to all methods.
5. Two-machine version drift. The RTX 5060 is **not yet audited**, so the cu128 stack is unverified there. Pin versions and log the host in every run.
6. Laptop disk is tight: D: has 5.9 GB free after the venv, and C: (repo) 4.1 GB. The checkpoints (1.77 GB) plus a ~1k-image subset must fit.
7. ~~No GitHub remote~~ Resolved 2026-10-20: `origin` configured and pushed. Large files (data, checkpoints, predictions) are still not backed up off-machine.
8. [FT] literature facts were extracted by LLM-assisted querying, not a line-by-line read. Re-verify numbers before quoting them.
9. Determinism is best-effort (`warn_only=True`); some CUDA ops are nondeterministic.

## 12. How to resume (for any agent)
1. Read this file fully, then the latest `reports/M*_*.md`.
2. Check §7 for the current module and §8 for frozen decisions. Do not silently change a frozen decision; add a new row instead.
3. Environment: on the dev laptop, activate `D:\aerial-ovd-store\venv`. On a new machine, create a venv from `requirements.txt` (CPU) or `requirements-gpu.txt` (cu128), set `AOVD_STORE`, run `python scripts/env_audit.py`, and commit the env JSON + lock file.
4. Follow the rules in §3. After finishing a phase, run the **context-keeper** agent (`.claude/agents/context-keeper.md`). It updates this file from repo facts only (§4–§8, §13, §14) and does not commit; the lead reviews and commits.

## 13. Change log
| Date | Phase | Change |
|---|---|---|
| 2026-10-20 | M0/M1 | Initial audit of repo and hardware; literature search; plan approved; git init; directory layout created; CONTEXT.md created |
| 2026-10-20 | M0+M1 end (after 1256354) | Context-keeper update: GPU corrected to RTX 3050 Ti; venv, versions, disk, and checkpoint commits recorded (§4); CastDet/OpenRSD/SAS/RT-OVAD verified + citation sweep (§5); H0 falsified, 251/256 (§6); module statuses (§7); D10–D12 added (§8); E0 full vocabulary = single prompt (§10); risks updated (§11); context-keeper agent noted (§9, §12); §14 phase log added |
| 2026-10-20 | M1 | GitHub remote `origin` added and `main` pushed (§4, §9, §11 risk 7, §14) |

## 14. Phase log
| Phase | Commit | Key commands | Outcome |
|---|---|---|---|
| M0 literature audit | `1256354` | Firecrawl web searches; Semantic Scholar citation sweep (2506.00333, 2601.22164); `python scripts/check_prompt_length.py --classes configs/lae80c_classes_raw.txt` | 13-work matrix; gap holds as of 2026-10-20; H0 falsified (251/256 tokens); gated decision recorded |
| M1 repo & compute audit | `1256354` | `python -m venv D:/aerial-ovd-store/venv`; pip install (torch 2.7.1 CPU wheels, transformers 4.57.3, …); `pip install -e .`; `pytest`; `python scripts/env_audit.py`; `python scripts/fetch_checkpoints.py --dry-run` | Laptop env recorded (RTX 3050 Ti, CPU torch); 16/16 tests pass; checkpoint commits resolved (1.77 GB, not downloaded). Pending: RTX 5060 audit, GitHub remote |
| Context-keeper update + fixes | (this commit) | context-keeper agent; M0 exit-criteria table added; README fixed; env audit re-run at 1256354 | M0 ✅; `.claude/agents/context-keeper.md` and `.gitattributes` committed |
| GitHub remote | (this commit) | `git remote add origin https://github.com/Himanshu-RG/Vision-Aerial-Imagery.git`; `git push -u origin main` | Repo on GitHub; RTX 5060 machine can now clone it |
