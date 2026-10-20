# CONTEXT.md — Project Ground Truth

> **Single source of truth for this project.** Any agent or person resuming work must read this file first.
> It is updated at the end of every phase (module). If this file conflicts with memory, chat history, or other docs, **this file wins** — unless it is stale, in which case fix it first.
> Last updated: 2026-10-20 · Phase: **M0 + M1 in progress**

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

## 4. Compute & environment (verified 2026-10-20)
| | Dev laptop (this machine) | Experiment machine |
|---|---|---|
| Role | coding, unit tests, CPU smoke tests on a small subset | **all real experiments + all latency measurements** |
| GPU | RTX 3050 Laptop, **4 GB** VRAM (shared with games; 1.8 GB in use at audit) | **RTX 5060, 8 GB** VRAM (Blackwell, sm_120) |
| Disk | C: 3.7 GB free, D: 7.6 GB free | ~100 GB free |
| CPU / RAM | i7-11800H 8C / 15.7 GB | TBD (run `scripts/env_audit.py`) |
| OS | Windows 11 Home 10.0.26300 | TBD |
| Python | 3.11.9 (system); system torch 2.2.2+**cpu** | TBD |

- **Sync:** via GitHub. The remote URL is not yet provided, and nothing is pushed until the user gives one.
- **Pinned stack (both machines):** torch 2.7.1 / torchvision 0.22.1. Use the CPU wheels on the laptop and the **cu128** wheels on the RTX 5060 (Blackwell needs torch ≥ 2.7 + CUDA 12.8). transformers 4.57.3. Exact locks go in `requirements-lock-<host>.txt`.
- **Large-file store:** `D:\aerial-ovd-store` on the laptop, which is outside OneDrive so large files don't sync. Its location is set by the `AOVD_STORE` env var or by `configs/paths.local.yaml` (gitignored).
- **Laptop data policy:** keep only a stratified subset (~1k dev images) of LAE-80C locally. Delete the zip after extraction.

## 5. Literature state (Module 0) — verified facts only
| Work | Status / key facts | Relevance |
|---|---|---|
| **Grounding DINO** (Liu et al., ECCV 2024; arXiv 2303.05499; github IDEA-Research/GroundingDINO) | Detector used here. Text encoder (BERT) is capped at **256 tokens**. HF checkpoint `IDEA-Research/grounding-dino-tiny` (689 MB safetensors). | Primary detector |
| **VocAda** (Liu, Hayes, Mancini, Ricci, Volpi, Csurka; ICIP 2025; arXiv 2506.00333; github OatmealLiu/VocAda) | Training-free pipeline: LLaVA-Next-7B caption → spaCy noun phrases → class selection by CLIP-text top-k (k=1 default) or by LLM. Detectors: Detic, VLDet, CoDet (**not GDINO**). Data: COCO OVE, Objects365 CDTE (**not aerial**). Gains up to +3.3 AP (COCO) and +2.2 AP (O365). Overhead 5.6 s/img with 17 GB (CLIP variant) or 10.7 s/img with 31 GB (LLM variant) on a V100. | Closest method; baseline E5 ("VocAda-style", not a reproduction) |
| **Tsourveloudis, "Do Open-Vocabulary Detectors Transfer to Aerial Imagery? A Comparative Evaluation"** (arXiv 2601.22164, Jan 2026) | Zero-shot evaluation on LAE-80C of GDINO, OWLv2, YOLO-World, YOLOE, and LLMDet. **Oracle per-image vocabulary (~3.2 classes) gives ~15× higher F1 than the 80-class prompt** (0.5% → 7.4%). Prompt engineering ("Aerial view of", synonyms) does not help. Metrics are P/R/F1 with IoA 0.7, not COCO mAP. **No deployable selector.** | Closest study; motivates headroom; our E0 replicates its direction with mAP |
| **LAE-DINO / LAE-1M** (Pan et al., AAAI 2025; arXiv 2408.09110; github jaychempan/LAE-DINO; HF `jaychempan/LAE-1M`, MIT) | Dynamic Vocabulary Construction is a **training-time** mechanism. LAE-80C benchmark: 80 classes, COCO JSON, images.zip 1.97 GB, built from DOTA-v2 / DIOR / FAIR1M / xView val splits. | Primary dataset source |
| **TagCLIP** (Lin et al., AAAI 2024; arXiv 2312.12828; github linyq2117/TagCLIP) | Training-free **local-to-global** CLIP multi-label classification for natural images. | Direct overlap with the global–local idea; must be a baseline |
| **FACTOR** (Zhao et al., ICML 2026; arXiv 2605.03294) | Training-free counterfactual TTA that corrects GDINO logits under corruptions. Does not select vocabulary. Not aerial. | Adjacent (OVOD test-time adaptation) |
| **Li & Tang, CMC 2026** ("Adapting GroundingDINO for RS detection… concise category-aware prompts") | Fine-tunes GDINO on DOTA/DIOR. Finds that concise 2–3-token prompts work best. No per-image selection. | Prompt-naming evidence |
| **Zhao et al., CVPR 2024** (Scene-adaptive & Region-aware Multi-modal Prompt; pp. 16741–16750) | Venue verified via CVF/dblp. Method details pending. | Scene-adaptive prompting (training-based) |
| CastDet (ECCV 2024, arXiv 2311.11646), OpenRSD (ICCV 2025), RT-OVAD (arXiv 2408.12246) | **Pending primary-source verification.** | Aerial OVD context |

**Draft gap statement:** Prior work has shown the oracle headroom of vocabulary reduction for aerial OVD. It has not tested whether a *deployable, inference-time* selector recovers that headroom for Grounding DINO, measured with COCO metrics, small- and rare-object breakdowns, and end-to-end latency.

**Novelty risk:** global CLIP selection is HIGH risk and is used only as a baseline. Global–local evidence is MEDIUM-HIGH risk (TagCLIP). Uncertainty-aware retention is MEDIUM risk (conformal multi-label prediction exists). An adaptive budget is HIGH risk as a standalone contribution, so it is framed as part of the selector.
**Realistic venue:** an EarthVision-type workshop, IGARSS, GRSL, or JSTARS, unless effects are large and robust.

## 6. Hypotheses (falsifiable)
- **H0 (confound):** the 80 class names exceed GDINO's 256-token limit, so the full vocabulary needs chunked prompts. Part of the prior study's collapse may be truncation.
- **H1:** oracle vocabulary > full vocabulary in mAP@50 on dev, with a bootstrap CI excluding 0 (the headroom).
- **H2:** some deployable global selector at a matched budget beats the full vocabulary.
- **H3:** local evidence raises vocabulary recall for small-object classes at a matched mean vocabulary size.
- **H4:** uncertainty-aware retention raises detection recall at a matched mean vocabulary size.
- **H5:** an adaptive budget Pareto-dominates fixed-k.
- **H6:** end-to-end latency, including the selector, is below the full chunked vocabulary.
- **Gate:** M6/M7 proceed only if M4/M5 show a selector beating the full vocabulary on dev. Otherwise pivot to an analysis paper.

## 7. Module roadmap & status
| Module | Description | Status |
|---|---|---|
| M0 | Literature audit & novelty | 🟡 in progress |
| M1 | Repo & compute audit | 🟡 in progress (git init + dirs done) |
| M2 | Dataset selection & validation, frozen protocol | ⬜ |
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
| M13 | Reproducibility & tests | ⬜ (cross-cutting) |
| M14 | Paper / publication readiness | ⬜ |

Dependency chain: M0 → M2. M1 → M2 → M3 → M4 → M5 → M6 → M7 → M10/M11 → M14.

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

## 9. Repository layout
```
vision/                      (git repo root, branch main)
  CONTEXT.md                 ← this file (ground truth)
  configs/                   YAML configs; paths.example.yaml (committed), paths.local.yaml (ignored)
  data/manifests/            split ID lists, dataset manifests (committed, small)
  src/aerial_ovd/{data,detector,vocabulary,evaluation,experiments,utils}/
  scripts/                   env_audit.py, fetch_checkpoints.py, ...
  tests/
  outputs/{logs,predictions,metrics,figures,env}/   (predictions ignored; env + metrics committed)
  reports/                   per-module reports (M0_*.md, M1_*.md, ...)
```

## 10. First experiment queued (after M0/M1): E0 headroom check
~200 stratified dev images on the RTX 5060, GDINO-T fp16. Conditions: (a) full 80-class vocabulary (chunked), (b) **oracle** per-image GT vocabulary (diagnostic), (c) random vocabulary size-matched to the oracle. Metrics: mAP@50, mAP@50:95, recall@100, latency, peak VRAM, failures.
- **Success:** 0 unexplained failures; pycocotools passes the known-answer test (GT as predictions gives AP = 1.0); oracle > full with a CI excluding 0.
- **Rethink if:** oracle ≈ full (no headroom), or oracle mAP@50 ≈ 0 (detector too weak, so address tiling first).

## 11. Risks / open issues
1. GDINO-T may be near the floor on aerial images (the prior study saw ~4% recall even with an oracle vocabulary). Tiling may be needed, applied identically to all methods.
2. 256-token limit / chunking (H0).
3. Two-machine version drift. Pin versions and log the host in every run.
4. Laptop disk is tight (D: 7.6 GB).

## 12. How to resume (for any agent)
1. Read this file fully, then the latest `reports/M*_*.md`.
2. Check §7 for the current module and §8 for frozen decisions. Do not silently change a frozen decision; add a new row instead.
3. Environment: create a venv from `requirements.txt` (CPU) or `requirements-gpu.txt` (cu128), set `AOVD_STORE`, and run `python scripts/env_audit.py`.
4. Follow the rules in §3. After finishing a phase, update this file (§5–§8, §13).

## 13. Change log
| Date | Phase | Change |
|---|---|---|
| 2026-10-20 | M0/M1 | Initial audit of repo and hardware; literature search; plan approved; git init; directory layout created; CONTEXT.md created |
