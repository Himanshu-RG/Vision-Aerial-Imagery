# CONTEXT.md — Project Ground Truth

> **Single source of truth for this project.** Any agent or person resuming work must read this file first.
> It is updated at the end of every phase (module). If this file conflicts with memory, chat history, or other docs, **this file wins** — unless it is stale, in which case fix it first.
> Last updated: 2026-10-24 · Phase: **M0 ✅ + M1 ✅ + M2 ✅ · M3 🟡 (pipeline built, CPU smoke OK) + M9 🟡 (evaluator validated) · next: run E0 on the RTX 5060 (GPU smoke first; runbook §12 step 5)**. No experimental results exist yet.

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

## 4. Compute & environment (both machines audited; see `reports/M1_environment.md`)
| | Dev laptop `LAPTOP-8N8B1VIP` (audited) | Experiment machine `LAPTOP-5CPRTI27` (audited; **reference environment**, D13) |
|---|---|---|
| Role | coding, unit tests, CPU smoke tests on a small subset | **all real experiments + all latency measurements** |
| GPU | **NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB** (driver 581.08; 2 MB in use at the 2026-10-24 10:10 UTC re-audit). Deliberately **unused**: CPU torch only (D12) | **NVIDIA GeForce RTX 5060 Laptop GPU**, compute capability **12.0** (Blackwell), 7.93 GB (torch) / 8151 MB (nvidia-smi), driver **616.92**. At the 2026-10-24 re-audit: torch 6.88 GB free, nvidia-smi 1665/8151 MB used (the first audit's 7081 MB is explained in §11 risk 10) |
| Disk free | C: (repo, OneDrive) **6.3 GB** · D: (store) **5.5 GB** (re-audit 2026-10-24 10:10 UTC, *before* the 0.69 GB GDINO-T download at 10:17 UTC; not re-measured since) | repo 194.2 GB · store 189.4 GB (re-audit 2026-10-24) |
| CPU / RAM | i7-11800H (env_audit reports 8 logical CPUs) / 15.7 GB | Intel64 Family 6 Model 198 (20 logical CPUs) / 15.4 GB |
| OS | Windows 11 Home, build 10.0.26300 (Python reports `Windows-10-10.0.26300`) | Windows build 10.0.26200 (Python reports `Windows-10-10.0.26200`) |
| Python / venv | **3.11.9**; venv **`D:\aerial-ovd-store\venv`** (system torch 2.2.2+cpu is not used) | **3.10.11**; venv `D:\aerial-ovd-store\venv` |
| torch / CUDA | torch 2.7.1+cpu, torchvision 0.22.1+cpu; no CUDA | torch **2.7.1+cu128**, torchvision 0.22.1+cu128, CUDA build 12.8, `cuda_available: true`, cuDNN 90701 |
| Shared exact pins (both) | transformers 4.57.3, huggingface_hub 0.36.2, safetensors 0.8.0, open_clip_torch 3.3.0, pycocotools 2.0.11, **numpy 2.2.6**, pillow 12.3.0, pyyaml 6.0.3, pydantic 2.14.1, **pandas 2.3.3**, **matplotlib 3.10.9**, psutil 7.2.2, pytest 9.1.1 | same |
| Records | `outputs/env/LAPTOP-8N8B1VIP.json` (re-audited 2026-10-24 at b95a7be, dirty tree), `requirements-lock-LAPTOP-8N8B1VIP.txt` (51 packages) | `outputs/env/LAPTOP-5CPRTI27.json` (re-audited 2026-10-24 at 6b70fe3, clean tree; interpreter `D:\aerial-ovd-store\venv`, Python 3.10.11), `requirements-lock-LAPTOP-5CPRTI27.txt` (53 packages; UTF-8 since 1bc146a) |
| Tests | **39 passed** at M3 (2026-10-24: 16 utility + 7 data + 16 M3 evaluator/detector/conditions). M2: 23; M1: 16/16 | **23 passed** (2026-10-24, pasted by the user) but in a **Python 3.11.9 venv**, not the reference `D:\aerial-ovd-store\venv` (Python 3.10.11, cu128). Pytest in the reference env is still unrecorded. The 16 M3 tests have not run there yet (the M3 code is not committed yet) |

- **Sync:** via GitHub, remote `origin` = https://github.com/Himanshu-RG/Vision-Aerial-Imagery.git (branch `main`, first pushed 2026-10-20). Commits use each machine's global git identity: `Himanshu-RG` on the dev laptop, `Amit-adh` on the experiment machine (commit 2e26ba2).
- **Pinned stack (both machines):** torch 2.7.1 / torchvision 0.22.1. Use the CPU wheels on the laptop and the **cu128** wheels on the RTX 5060 (Blackwell needs torch ≥ 2.7 + CUDA 12.8). `requirements.txt` now holds **exact pins** for every package, aligned to the RTX 5060 reference env (D13). `requirements-gpu.txt` (cu128) includes `requirements.txt` via `-r`. Exact locks are in `requirements-lock-<host>.txt`.
- **Version alignment (M1, 2026-10-21):** the RTX 5060 (Python 3.10) had resolved numpy 2.2.6 / pandas 2.3.3 / matplotlib 3.10.9, while the laptop had 2.4.6 / 3.0.6 / 3.11.2. The laptop was reinstalled to the RTX 5060 versions. **Accepted residual differences:** the torch build (+cpu vs +cu128); the Python minor version (3.10 vs 3.11); the py3.10 backports `exceptiongroup`/`tomli` (RTX 5060 only); minor transitive versions of contourpy, fonttools, networkx, wcwidth.
- **Checkpoints** (`configs/checkpoints.yaml`; the 2026-10-20 dry-run resolved these commits; the sha256 lock `configs/checkpoints.lock.json` is written on the first real download, D11):
  - `IDEA-Research/grounding-dino-tiny` **downloaded and locked (M3)** at commit **`a2bb814dd30d776dcf7e30523b00659f4f141c71`**, fetched 2026-10-24 10:17 UTC on the dev laptop. Lock has sha256 + size of all 8 files, e.g. `model.safetensors` sha256 `1a2412ef…42f3` (689,359,096 bytes). The **RTX 5060 has not fetched it yet**: `fetch_checkpoints.py` re-uses the locked commit and aborts on any sha256 mismatch.
  - `chendelong/RemoteCLIP@bf1d8a3ccf…` (577.2 MB): not downloaded yet.
  - `timm/vit_base_patch32_clip_224.openai@a6f597a30f…` (577.2 MB): not downloaded yet. Both CLIP encoders are selector encoders (D7) for M4.
  - Dry-run total: 1.77 GB.
- **Large-file store:** `D:\aerial-ovd-store` on the laptop, which is outside OneDrive so large files don't sync. Its location is set by the `AOVD_STORE` env var or by `configs/paths.local.yaml` (gitignored). Set `HF_HOME=<store>/hf_cache`.
- **Laptop data policy:** keep only the **dev split** (730 images, 399 MB at `<store>/datasets/LAE-80C-images`) of LAE-80C locally; `images.zip` was deleted after extraction (M2). The GDINO-T checkpoint is in `<store>/checkpoints/grounding-dino-tiny` (M3). Disk free at the 2026-10-24 re-audit: see the table above.
- **RTX 5060 data:** all 3,592 images, fetched with `scripts/fetch_lae80c_images.py --split all --delete-zip` **done 2026-10-24: 3,592/3,592 OK** (`data/manifests/lae80c_images_check_LAPTOP-5CPRTI27_all.json`; zip sha256 identical to the laptop's).

## 5. Literature state (Module 0) — verified facts only
Full matrix with 13 works and verification levels ([FT] full text via LLM-assisted query, [AB] abstract/landing page, [META] metadata only): `reports/M0_literature_matrix.md`. Re-check exact numbers before quoting them in the paper.

| Work | Status / key facts | Relevance |
|---|---|---|
| **Grounding DINO** (Liu et al., ECCV 2024; arXiv 2303.05499; github IDEA-Research/GroundingDINO) [META+repo] | Detector used here. BERT text encoder, `max_text_len` = **256** in the grounding-dino-tiny config. HF checkpoint `IDEA-Research/grounding-dino-tiny`. | Primary detector |
| **VocAda** (Liu, Hayes, Mancini, Ricci, Volpi, Csurka; ICIP 2025, pp. 540–545; arXiv 2506.00333; github OatmealLiu/VocAda) [FT] | Training-free pipeline: LLaVA-Next-7B caption → spaCy noun phrases → class selection by CLIP-text top-k (k=1 default) or by LLM. Detectors: Detic, VLDet, CoDet (**not GDINO**). Data: COCO OVE, Objects365 CDTE (**not aerial**). Gains up to +3.3 AP (COCO) and +2.2 AP (O365). Overhead 5.6 s/img with 17 GB (CLIP variant) or 10.7 s/img with 31 GB (LLM variant) on a V100. Retains 94.8% (CLIP) and 86.1% (LLM) of relevant classes. | Closest method; baseline E5 ("VocAda-style", not a reproduction) |
| **Tsourveloudis, "Do Open-Vocabulary Detectors Transfer to Aerial Imagery? A Comparative Evaluation"** (arXiv 2601.22164, Jan 2026) [FT] | Zero-shot evaluation on LAE-80C of GDINO, OWLv2, YOLO-World, YOLOE, and LLMDet. **Oracle per-image vocabulary (~3.2 classes) gives ~15× higher F1 than the 80-class prompt** (0.5% → 7.4%). (Our M2 count from the LAE-80C JSON is **2.24** classes/image; the discrepancy is unexplained, §11. Our secondary, standard all-80-class evaluation (D15) is the one meant for comparability with this paper.) Prompt engineering ("Aerial view of", synonyms) does not help. Metrics are P/R/F1 with IoA 0.7, not COCO mAP. **No deployable selector.** It **explicitly calls for "intelligent vocabulary filtering" as future work.** It does not discuss the 256-token limit. | Closest study; motivates headroom; our E0 replicates its direction with mAP |
| **LAE-DINO / LAE-1M** (Pan et al., AAAI 2025, 39(6):6281–6289; arXiv 2408.09110; github jaychempan/LAE-DINO; HF `jaychempan/LAE-1M`, MIT) [AB+repo+HF] | Dynamic Vocabulary Construction is a **training-time** mechanism. LAE-80C benchmark: 80 classes, COCO JSON, images.zip 1.97 GB, built from DOTA-v2 / DIOR / FAIR1M / xView val splits. **Image count resolved in M2: 3,592 images, 86,558 boxes** (matches the citing paper). The HF README's "~20k (val)" does not describe this file. Facts from the file: §5a. | Primary dataset source |
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

**Pending pre-submission checks** (matrix §4): full-text read of CastDet's protocol (which classes are novel per dataset); RT-OVAD venue and exact GDINO baseline setting; PISA content; conformal multi-label prediction references for M6; ~~the exact LAE-80C image count (M2)~~ resolved in M2 (3,592); the reason for the 2.24 vs. 3.2 classes/image difference from 2601.22164 (open).

### 5a. LAE-80C dataset facts (Module 2; `reports/M2_dataset.md`, `data/manifests/lae80c_prepare_report.json`)
- **Source:** HF `jaychempan/LAE-1M` at commit **`60ce78a91a09ea9975eba6de18ffe621936dbc0f`** (MIT per the dataset card). Annotation file `LAE-80C/LAE-80C-benchmark.json`, sha256 `7e848c531a0bdfdfea58ea3db298ad7db697084a3c5bd98fa4f353b316599fb2`. `LAE-80C/images.zip` sha256 `288988fe25a3d7173c5085d0660b785bdc6a819559db3d7af78e60a1ee7cb398`, 3,592 members, all in the annotations, no extras.
- **Size:** 3,592 images; 86,558 boxes raw → **86,539 after cleaning** (D17). Horizontal boxes only (all 45,375 xView polygons are axis-aligned and match their bbox within 1 px; any OBB→HBB conversion was done upstream by the LAE-1M authors, none by us).
- **Five sources with disjoint label spaces.** Every image is annotated only for its own source's classes:

  | Source | Images | Class ids | Boxes/img | Small boxes | Classes/img | dev / test |
  |---|---|---|---|---|---|---|
  | FAIR-18C | 2,130 | 25–42 (18) | 18.0 | 91% | 2.50 | 435 / 1,695 |
  | Xview-34C | 758 | 47–80 (34) | 59.9 | 34% | 2.29 | 151 / 607 |
  | DIOR-18C | 536 | 1–18 (18) | 4.0 | 30% | 1.43 | 108 / 428 |
  | DOTA-6C | 137 | 19–24 (6) | 3.8 | 17% | 1.07 | 28 / 109 |
  | PP-4C | 31 | 43–46 (4) | 4.8 | 3% | 2.71 | 8 / 23 |

  The same concept appears under different names in different sources (`vehicle` DIOR / `Small Car` FAIR / `Passenger Car` xView; `airplane` DIOR / `Fixed-wing Aircraft`, `Small Aircraft` xView). Under standard COCO evaluation, a correct detection under another source's name is a false positive (→ D15, H7).
- **Image size identifies the source:** 600² always FAIR, 800² always DIOR, 1024² xView or DOTA, irregular (~1052–1259 × ~683–754) PP. This is a shortcut; selectors must never see it (D16).
- **Statistics (after cleaning):** size share 59.3% small / 35.1% medium / 5.7% large (COCO bins 32², 96²). Boxes/image mean 24.1, max 668. Classes/image mean **2.24** (dev 2.21, test 2.25). Dev small-box share 62.3% (test 58.5%).
- **Long tail:** `helipad` is in **1 image** (test only; missing from dev). `container crane`, `Flat Car`, `Straddle Carrier`, `Reach Stacker` are in 3–5 images each. **16 classes have only 1–2 dev images.** Per-class AP for these is not meaningful; rare classes are pooled by frequency band (D15).
- **Image validation:** dev laptop, dev split **730/730 OK** (0 missing, 0 unreadable, 0 size mismatches; `data/manifests/lae80c_images_check_LAPTOP-8N8B1VIP_dev.json`). RTX 5060, all images: **3,592/3,592 OK** (same zip sha256 `288988fe…`).

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
  - **Caveat:** the margin is only **5 tokens**. Any prefix ("aerial view of"), synonym expansion, or longer naming policy will exceed 256. **Re-check whenever the naming policy changes.** Whether the prior study's prompt variants were truncated is unverified.
  - **Re-check (M2, 2026-10-24):** with the frozen "light" naming policy (D14) the full prompt is **249 / 256 tokens**, still a single prompt (D10). The exact prompt string is stored in `data/manifests/lae80c_prepare_report.json` (`stats.full_prompt_light_names`).
  - **Re-check (M3, 2026-10-24):**
    - The prompt now uses the **official Grounding DINO caption format** `"name1 . name2 . … ."` (D20) in `names.build_prompt`, `check_prompt_length.py` and the detector.
    - Token counts are unchanged: light 249, raw 251.
    - `prepare_lae80c.py` was re-run. Only `full_prompt_light_names` in the prepare report changed; the split is unchanged.
- **H1:** oracle vocabulary > full vocabulary in mAP@50 on dev, with a bootstrap CI excluding 0 (the headroom).
- **H2:** some deployable global selector at a matched budget beats the full vocabulary.
- **H3:** local evidence raises vocabulary recall for small-object classes at a matched mean vocabulary size.
- **H4:** uncertainty-aware retention raises detection recall at a matched mean vocabulary size.
- **H5:** an adaptive budget Pareto-dominates fixed-k.
- **H6:** end-to-end latency, including the selector, is below that of the full vocabulary. For raw names that is a single prompt, not chunked (see H0).
- **H7 (added M2; `reports/M2_dataset.md` "New hypothesis"):** a substantial share of the full-vs-oracle gap under standard evaluation comes from **cross-source label-space mismatch** (§5a).
  - **Prediction:** the gap shrinks under source-aware evaluation (D15), and the source-label-space oracle recovers much of the per-image-oracle gain.
  - **Falsified if** the gap under source-aware evaluation stays within the bootstrap CI of the standard-evaluation gap.
- **Gate:** M6/M7 proceed only if M4/M5 show a selector beating the full vocabulary on dev (H2 holds). If H1 fails, pivot to an analysis paper. If H1 holds but H2 fails, report the negative result and analyse selection errors before M5.

## 7. Module roadmap & status
| Module | Description | Status |
|---|---|---|
| M0 | Literature audit & novelty | ✅ exit criteria met (`reports/M0_gap_hypotheses_novelty.md` §5). Pre-submission re-checks remain open (§5) |
| M1 | Repo & compute audit | ✅ exit criteria met on both machines (`reports/M1_environment.md` "Exit criteria"). Open item: pytest in the RTX 5060 *reference* env (D13) not yet recorded (a 3.11.9 venv there passed 23/23) |
| M2 | Dataset selection & validation, frozen protocol | ✅ exit criteria met (`reports/M2_dataset.md` "Exit criteria"): manifest, naming policy (D14), evaluation protocol defined (D15; implemented in M9), split frozen (D18). RTX 5060 full image validation **done** (3,592/3,592 OK) |
| M3 | Grounding DINO baseline | 🟡 **pipeline ✅, GPU smoke + E0 pending** (`reports/M3_baseline.md` "Exit criteria"). Done: reproducible inference (pinned checkpoint + lock, config-hash run id, provenance), saved predictions, CPU smoke test on the laptop. Pending: GPU smoke on the RTX 5060, then baseline numbers (E0) |
| M4 | Master vocab + simple baselines (A full, B fixed-reduced, C random, D global CLIP, E VocAda-style) | ⬜ |
| M5 | Global–local relevance | ⬜ |
| M6 | Uncertainty-aware retention | ⬜ (gated) |
| M7 | Adaptive budget | ⬜ (gated) |
| M8 | Unified experiment runner (E1–E8) | ⬜ (grows with M3–M4; the M3 config-driven `scripts/run_detection.py` + `scripts/evaluate_run.py` are its starting point) |
| M9 | Metrics + validation on known examples | 🟡 **evaluator validated with known-answer tests + bootstrap; P/R/F1 operating point pending**. Done: the standard and source-aware evaluators (D15), the known-answer tests (`reports/M3_baseline.md` "Evaluation"), and the paired bootstrap (fast AP50 matches COCOeval to 1e-9). The real-data fast-vs-COCOeval check passed on the CPU smoke run. Pending: P/R/F1 at an operating point chosen on dev. Bootstrap CIs cover AP50@900 only |
| M10 | Failure analysis | ⬜ |
| M11 | Cross-dataset generalization | ⬜ (VisDrone mapping deferred here from M2, D19) |
| M12 | Efficiency / resume / profiling | ⬜ (cross-cutting) |
| M13 | Reproducibility & tests | ⬜ (cross-cutting; 39 tests exist: 16 utility + 7 data/split/naming + 16 M3 evaluator/detector/conditions) |
| M14 | Paper / publication readiness | ⬜ |

Dependency chain: M0 → M2. M1 → M2 → M3 → M4 → M5 → M6 → M7 → M10/M11 → M14. M9 runs alongside M3 and gates trust in its numbers.

**M2 to-do (from `reports/M1_environment.md`): done.** Image count resolved (3,592); manifest built; boxes and categories validated and cleaned (D17); split frozen (D18); naming policy "light" + token re-check 249/256 (D14); VisDrone mapping deferred to M11 (D19).

**M3 + M9 to-do** (from `reports/M2_dataset.md` "Next"; status from `reports/M3_baseline.md`):
0. ~~RTX 5060 full image fetch~~ done 2026-10-24 (3,592/3,592 OK).
1. ~~Fetch the GDINO-T checkpoint~~ done on the dev laptop; locked (§4). The RTX 5060 still has to fetch it (§12 step 5).
2. ~~Detector inference with positive-map class scoring~~ done: `src/aerial_ovd/detector/gdino.py`, mean over tokens (D20 supersedes D5), top-900 (D21).
3. ~~Source-aware and standard COCO evaluators with known-answer tests~~ done: `src/aerial_ovd/evaluation/coco_eval.py`, all known-answer tests pass.
4. ~~CPU smoke test on the laptop~~ done. It was a pipeline check only and is not a result: 3 images × 4 conditions all `ok`, 900 detections each. The sampler rounding bug it exposed is fixed and tested.
5. **Next:** on the RTX 5060, run pytest in the reference env, then the GPU smoke test, then E0 (§10), then evaluate and commit the metrics. Runbook: §12 step 5.
6. After E0: decide the gates H1 (headroom) and H7 (label-space share) (§6), then go to M4.
7. M9 remainder: P/R/F1 at an operating point chosen on dev (D15).

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
| D13 | The RTX 5060 machine `LAPTOP-5CPRTI27` is the **reference environment**. `requirements.txt` exact pins follow its versions. Any future dependency change must be installed on **both** machines, and **both** lock files (`requirements-lock-<host>.txt`) and env audits refreshed | Version drift found in M1 (numpy/pandas/matplotlib; pandas 3.x changes defaults); all reported results come from this machine (D9) | 2026-10-21 |
| D14 | **Naming policy "light"** (`src/aerial_ovd/vocabulary/names.py`): master vocabulary = all 80 LAE-80C categories; evaluation always uses category ids. Prompt names: lower-case everything; split the 7 glued DIOR names (`groundtrackfield` → `ground track field`, `baseballfield`, `basketballcourt`, `storagetank`, `tenniscourt`, `trainstation`, `golffield`); nothing else (no synonyms, no prefixes, slashes kept, overlapping classes kept as defined). Same names for every method. Full prompt **249 / 256 tokens** → single prompt (D10). Mapping in `data/manifests/lae80c_vocabulary.csv` | Glued names hurt text grounding (§11 risk 4); keep within the token limit; concise names (Li & Tang, §5) | 2026-10-24 |
| D15 | **Evaluation protocol** (frozen now, implemented and validated in M9). **Primary: source-aware ("federated") COCO eval**: per image, only predictions whose class is in the image's source label space are scored; other classes are ignored (neither TP nor FP). Source identity is used only by the evaluator, never at inference. **Secondary: standard all-80-class COCO eval** (comparability with 2601.22164). Metrics: mAP@50 (headline), mAP@50:95, AR@100; per-class AP with rare classes pooled by frequency band; P/R/F1 at a score threshold chosen on dev. **Two oracle diagnostics (never methods):** (i) per-image GT vocabulary; (ii) source-label-space vocabulary (the 6–34 classes of the image's source). **No novel-class claims** (GDINO pretraining covers concepts such as car, airplane, ship; a "novel" breakdown would need a pretraining-vocabulary overlap audit) | Disjoint label spaces (§5a); assumes exhaustive within-source annotation (unverified, §11) | 2026-10-24 |
| D16 | **Selector input rule:** vocabulary selectors receive only the decoded image pixels, resized/cropped by the selector's own fixed preprocessing. Forbidden: file name, source tag, original image size, EXIF/metadata, any annotation-derived information. M4 unit tests must check that the selector API does not accept these | Image size identifies the source (shortcut, §5a); rule 3 | 2026-10-24 |
| D17 | **Cleaning policy** (`configs/data_lae80c.yaml` `cleaning`): clip boxes to image bounds (**972 boxes**, all FAIR/DOTA tile-edge objects; ≥ 58.4% of each box's area kept; none fully outside); drop exact duplicates (**19**, all FAIR; same image, class, box rounded to 0.1 px); drop non-positive boxes (0 found); keep and flag boxes with a side < 2 px (**23**: 21 xView, 2 FAIR). 86,558 → 86,539 boxes. Every action logged in `data/manifests/lae80c_prepare_report.json` | Rule 7 (no silent fixes); minimal intervention | 2026-10-24 |
| D18 | **D2 executed** (refines D2's method): iterative multi-label stratification over class presence (Sechidis et al., 2011; `src/aerial_ovd/data/split.py`), **seed 2026**, dev fraction 0.20 → **dev 730 / test 2,862** (per source in §5a). ID lists `data/manifests/lae80c_dev_ids.txt` sha256 `131348ae32b1f97774ced5b5d7aa4e358ae1867b1da3ae9e1570ddd65b2b7f8c`, `data/manifests/lae80c_test_ids.txt` sha256 `1d6b2c398d3154c9b3ef58572931d0826ceb302f0a12f3f0d0e65380511dbd0e`. Committed in the M2 commit, before any detector run. `prepare_lae80c.py` refuses to write a different split unless `--allow-resplit` is passed, which requires a new decision row. `helipad` is absent from dev | Class-presence stratification also balances sources (label spaces are disjoint) and rare classes; no test leakage (rule 3) | 2026-10-24 |
| D19 | Refines D3: the VisDrone-DET class mapping is **deferred from M2 to M11**; it does not affect the LAE-80C protocol | `reports/M2_dataset.md` "Dataset choice" | 2026-10-24 |
| D20 | **Supersedes D5.** Class score = sigmoid(token logits) @ row-normalised positive map, i.e. the **mean** over the class name's tokens (official `PostProcessCocoGrounding`). `max` over tokens (D5) is kept only as an ablation (`class_reduce: max`). Caption in the official format `"name1 . name2 . … ."`: lower-case, light names (D14), category-id order for every condition. Token counts unchanged (light 249, raw 251) | Match the official Grounding DINO COCO evaluation (`demo/test_ap_on_coco.py`) | 2026-10-24 |
| D21 | Keep the **top-900** (query, class) pairs per image. No NMS and no score threshold for AP. **Headline metric: AP50 at maxDets 900.** AP / AP50 / AR at maxDets **100** are also reported, for comparability with the official protocol. Refines D15's metric list | GT density reaches 668 boxes/image (mean 24.1). A 100-detection cap would bound recall regardless of vocabulary. The official script uses 300 | 2026-10-24 |
| D22 | **E0 sample:** 200 dev images, exact-size class-stratified sample (iterative stratification, then a deterministic trim or top-up to exactly n), sample seed **2026**. Result: FAIR 119 / xView 41 / DIOR 30 / DOTA 8 / PP 2; 75 of the 79 dev classes covered; test never touched (unit-tested). **Latency** is measured per (image, condition) as preprocess (tokenise + positive map + resize/normalise) + forward + postprocess, with CUDA sync. It excludes disk I/O (`load_ms`) and model load (`model_load_s`), which are recorded separately. The 3 warm-up passes are not recorded | Frozen, reproducible headroom sample on dev only (rule 3); latency definition fixed before any timing (rule 9, D9) | 2026-10-24 |

## 9. Repository layout
```
vision/                      (git repo root, branch main; remote origin = github.com/Himanshu-RG/Vision-Aerial-Imagery)
  CONTEXT.md                 ← this file (ground truth)
  README.md, pyproject.toml, requirements.txt (CPU), requirements-gpu.txt (cu128), requirements-lock-<host>.txt
  .claude/agents/context-keeper.md   agent that updates this file after each phase
  configs/                   paths.example.yaml (committed), paths.local.yaml (ignored), checkpoints.yaml,
                             checkpoints.lock.json (M3: GDINO-T commit + sha256/size of all 8 files),
                             lae80c_classes_raw.txt (80 raw names), lae80c_classes_light.txt (80 light names, D14),
                             data_lae80c.yaml (M2: pinned HF revision, cleaning switches, split fraction/seed/method, size bins)
                             experiments/e0_headroom.yaml (M3: E0 config: sample, conditions, detector settings, §10)
  data/manifests/            (committed, small; written by scripts/prepare_lae80c.py unless noted)
                             lae80c_manifest.csv (3,592 rows: id, file, source, split, size, n_boxes, n_classes, classes, size-bin counts, tiny flag)
                             lae80c_dev_ids.txt / lae80c_test_ids.txt (FROZEN split, D18)
                             lae80c_vocabulary.csv (80 rows: id, raw name, prompt name, source, n_boxes, n_images_dev/test)
                             lae80c_prepare_report.json (cleaning log, stats, sha256 of inputs/outputs)
                             lae80c_images_check_<host>_<split>.json (from scripts/fetch_lae80c_images.py)
  src/aerial_ovd/{data,detector,vocabulary,evaluation,experiments,utils}/   (utils: seed, logging, config, run_id, paths)
                             data/lae80c.py (load_coco, source_of, source_label_space, clean_coco, build_manifest, subset_coco)
                             data/split.py (deterministic iterative stratification) · vocabulary/names.py (naming policies light/raw, official-format prompt builder)
                             detector/gdino.py (M3: frozen HF GroundingDINO wrapper; official caption + positive map, mean/max class reduce, top-k, per-stage timing)
                             vocabulary/conditions.py (M3: `full` (deployable) + oracle diagnostics `oracle_gt`, `oracle_source`, control `random_matched`)
                             evaluation/coco_eval.py (M3/M9: `evaluate` standard/source_aware, maxDets 100/300/900, per-size, per-class;
                                                      `per_image_matches`, `ap50_from_matches`, `paired_bootstrap`)
  scripts/                   env_audit.py (`--write-lock`), fetch_checkpoints.py (`--dry-run`, `--only`), check_prompt_length.py,
                             prepare_lae80c.py (clean → manifest → frozen split → vocabulary → report; refuses to resplit),
                             fetch_lae80c_images.py (pinned images.zip → extract a split → validate every image),
                             run_detection.py (M3: config-driven runner; exact-size stratified sample, resume, incremental JSONL,
                                               failure/OOM logging, warm-up, GPU snapshot at start/end, provenance),
                             evaluate_run.py (M3: both protocols, latency/memory summary, paired bootstrap vs `full`,
                                              fast-vs-COCOeval consistency check that aborts on mismatch)
  tests/                     39 unit tests (16 utility + test_data_lae80c.py: 7 + test_coco_eval.py and test_gdino_and_conditions.py: 16)
  outputs/{logs,predictions,metrics,figures,env}/   (predictions ignored; env + metrics committed)
                             predictions/<run_id>/: run_config.json, <condition>.jsonl, log.jsonl (run_id = <name>-<config hash>)
                             metrics/<run_id>/: metrics.json, run_config.json (committed)
  reports/                   per-module reports (M0_literature_matrix.md, M0_gap_hypotheses_novelty.md, M1_environment.md, M2_dataset.md,
                             M3_baseline.md, ...)
<store>/ (not committed)     hf_cache/, datasets/LAE-1M/ (raw HF download), datasets/LAE-80C-clean/{all,dev,test}.json (cleaned COCO GT;
                             sha256 in lae80c_prepare_report.json), datasets/LAE-80C-images/ (images), checkpoints/grounding-dino-tiny/ (M3)
```

## 10. First experiment: E0 headroom check (configured in M3; **not run yet**)
**Config:** `configs/experiments/e0_headroom.yaml`. Runbook commands: §12 step 5. Inference protocol: `reports/M3_baseline.md` "Inference protocol".
- **Sample:** 200 dev images from the frozen dev split (D18), exact-size class-stratified, sample seed 2026 (D22).
- **Hardware:** RTX 5060, reference env (D9, D13).
- **Detector settings:**
  - GDINO-T, **fp32** (fp16 is a separate later ablation);
  - "light" names (D14), official caption format;
  - mean class scoring (D20), top-900 (D21);
  - 3 warm-up passes; `seed: 0`.

**Conditions:**
- (a) `full`: the 80-class vocabulary as a **single prompt** (249 tokens, so no chunking, D10). This is the only deployable condition.
- (b) `oracle_gt`, **oracle diagnostic (i):** the per-image GT vocabulary.
- (b′) `oracle_source`, **oracle diagnostic (ii):** the source-label-space vocabulary (the 6–34 classes of the image's source, D15). Comparing (b) and (b′) splits the headroom into "which label space applies" and "which classes are in this image" (H7).
- (c) `random_matched`: random classes, as many as `oracle_gt`, with `random_seed: 0`. It uses the GT class count, so it is a size-matched **control**, not a method.

Every condition is scored under **both evaluations** (D15): primary source-aware (federated) COCO eval and secondary standard all-80-class COCO eval.

**Metrics:**
- AP50@900 (headline, D21), AP@900, AP75;
- AP / AP50 / AR at maxDets 100 (official-comparable);
- AP50 and R50 per COCO size bin, per-class AP50 / AP / R50;
- paired image-level bootstrap of the AP50@900 difference vs. `full` (1,000 resamples, seed 0);
- latency median / p90 (D22), peak VRAM, failures.

**Criteria:**
- **Success:** 0 unexplained failures and oracle > full with a CI excluding 0. Both evaluators already pass the known-answer tests (M9); `evaluate_run.py` also aborts if fast AP50 and COCOeval disagree on the real data.
- **Rethink if:** oracle ≈ full (no headroom), or oracle AP50 ≈ 0 (detector too weak, so address tiling first, §11 risk 17).

## 11. Risks / open issues
1. GDINO-T may be near the floor on aerial images (the prior study saw ~4% recall even with an oracle vocabulary). Tiling may be needed, applied identically to all methods.
2. 256-token limit: resolved for raw names (251/256) and for the frozen "light" names (**249/256**, D14), but the **7-token margin** means any prefix or synonym change can silently truncate. Always re-run the token check (D10).
3. ~~LAE-80C image count is unresolved~~ **Resolved in M2 (2026-10-24):** the annotation JSON has **3,592 images** (86,558 boxes), matching the citing paper. The HF README's "~20k (val)" does not describe this file.
4. **LAE-80C class names have naming problems:**
   - concatenated names (`groundtrackfield`, `storagetank`, …): **fixed** by D14 (7 DIOR names split);
   - mixed case: **fixed** by D14 (lower-case);
   - hierarchy / cross-source overlap (`vehicle` vs `Small Car` vs `Passenger Car`; `airplane` vs `Fixed-wing Aircraft`): **kept as defined** (D14). Handled at evaluation time by the source-aware protocol (D15); still a source of ambiguous "wrong-class" detections under the standard evaluation (risk 12).
5. Two-machine version drift. ~~RTX 5060 not audited~~ Audited (env JSON timestamp 2026-10-20 17:05 UTC; recorded in the M1 report 2026-10-21): cu128 stack verified (`cuda_available: true`). Drift in numpy/pandas/matplotlib was found and fixed by exact pins (D13). Residual accepted differences: torch build, Python 3.10 vs 3.11, py3.10 backports, minor transitive deps. Keep logging the host in every run, and follow D13 for any dependency change.
6. Laptop disk is tight. At the 2026-10-24 10:10 UTC re-audit, D: (store) had **5.5 GB** free and C: (repo) **6.3 GB**. Since then the GDINO-T checkpoint (0.69 GB) was downloaded, and disk free was not re-measured. The two CLIP checkpoints (577.2 MB each, dry-run) must still fit.
7. ~~No GitHub remote~~ Resolved 2026-10-20: `origin` configured and pushed. Large files (data, checkpoints, predictions) are still not backed up off-machine.
8. [FT] literature facts were extracted by LLM-assisted querying, not a line-by-line read. Re-verify numbers before quoting them.
9. Determinism is best-effort (`warn_only=True`); some CUDA ops are nondeterministic.
10. **RTX 5060 GPU-memory anomaly (largely explained):** at the first audit nvidia-smi reported 7081/8151 MB used; the 2026-10-24 re-audit shows 1665/8151 MB, so the earlier reading was another process using the GPU, not an accounting bug. Before any latency or memory run (D9), confirm the GPU is idle and record `torch.cuda.mem_get_info()` at run start. This is an **M12 requirement**. It is partly implemented in M3: `run_detection.py` records `gpu_at_start` / `gpu_at_end` in `run_config.json` (torch free/total, nvidia-smi used MB, util %, power). Confirming that the GPU is idle remains a manual step.
11. On the RTX 5060, `pytest` passed 23/23 in a **Python 3.11.9 venv** that is not the audited reference env (`D:\aerial-ovd-store\venv`, Python 3.10.11, cu128). Experiments must run in the reference env; re-run `pytest` there and record it before the first GPU experiment. Expect 39 tests after pulling M3. `run_detection.py` exits with code 2 if the config asks for CUDA and none is available (wrong venv).
12. **Disjoint label spaces + source-identifying image sizes (shortcut risk).** The five sources annotate disjoint class sets, and image size reveals the source (600² FAIR, 800² DIOR, 1024² xView/DOTA, irregular PP; §5a). A selector that sees size or metadata could "learn" the label space instead of the image content. Mitigation: D16 (pixels only; M4 tests on the selector API); evaluator-only use of the source (D15). Note: pixels at the selector's fixed preprocessing may still carry source cues (resolution, sensor style); this is not tested yet.
13. **Long tail:** `helipad` is in 1 image in the whole benchmark (test only, so absent from dev and untunable); 16 classes have only 1–2 dev images; `container crane`, `Flat Car`, `Straddle Carrier`, `Reach Stacker` are in 3–5 images overall. Per-class AP is unreliable; pool rare classes by frequency band (D15). `PP-4C` has only 31 images (8 in dev). In E0 (200 images) per-class AP covers only the classes present in the sample (75 of the 79 dev classes, D22), and PP has only 2 images there.
14. **Exhaustive within-source annotation is an unverified assumption.** Under the source-aware evaluation (D15), a correct detection of an unannotated object of an in-label-space class counts as a false positive; if a source is not exhaustively annotated, scores are biased downward. Standard for DIOR/DOTA/FAIR1M/xView, but not checked by us.
15. **Classes per image 2.24 (our count from the JSON) vs. ~3.2 in arXiv 2601.22164 is unexplained.** It affects the comparability of our oracle-vocabulary results with that paper. Possible causes are not investigated (TBD).
16. **HF-vs-official numerical equivalence is not verified.** HF transformers re-implements Grounding DINO. We assume it matches the official IDEA-Research code because both use the same checkpoint, preprocessing, caption format, and class scoring (D20). That has not been checked against the official code. Any comparison with numbers from the official repo must carry this caveat (`reports/M3_baseline.md` "Known limitations").
17. **Single resolution; 59% of objects are small.** E0 runs at one resolution (shortest edge 800, longest ≤ 1333). 59.3% of boxes are COCO-small (62.3% in dev, §5a), so recall on small objects may be bounded by resolution and not by vocabulary (compare risk 1). If tiling is added later, it must apply identically to all methods.

## 12. How to resume (for any agent)
1. Read this file fully, then the latest `reports/M*_*.md`.
2. Check §7 for the current module and §8 for frozen decisions. Do not silently change a frozen decision; add a new row instead.
3. Environment: on the dev laptop, activate `D:\aerial-ovd-store\venv`. On a new machine, create a venv from `requirements.txt` (CPU) or `requirements-gpu.txt` (cu128), set `AOVD_STORE`, run `python scripts/env_audit.py`, and commit the env JSON + lock file. Any dependency change follows D13: install on both machines and refresh both lock files and env audits.
4. **Data (after pulling M2):**
   - **RTX 5060 (done 2026-10-24, 3,592/3,592 OK):** `python scripts/fetch_lae80c_images.py --split all --delete-zip`. It downloads the pinned `images.zip`, extracts all 3,592 images to `<store>/datasets/LAE-80C-images`, validates every image against the committed manifest, and writes `data/manifests/lae80c_images_check_<host>_all.json`; commit that file. It needs ~1.97 GB + 1.5 GB free in the store. The cleaned GT (`<store>/datasets/LAE-80C-clean/*.json`) is not in git: regenerate it with `python scripts/prepare_lae80c.py` (it fetches the pinned annotation JSON if absent; `--download` forces a fetch) and check its sha256 against `data/manifests/lae80c_prepare_report.json`. The script refuses to change the frozen split.
   - **Dev laptop:** dev images only (`--split dev --delete-zip`, already done).
5. **E0 runbook (RTX 5060, after pulling M3).** Use the reference env `D:\aerial-ovd-store\venv` (D13) and keep the GPU otherwise idle (D9).
   1. Run `pytest` and record the result (expect 39 passed; §11 risk 11).
   2. Make sure `<store>/datasets/LAE-80C-clean/dev.json` exists, because `evaluate_run.py` reads it. If it is absent, run `python scripts/prepare_lae80c.py` (step 4).
   3. `python scripts/fetch_checkpoints.py --only grounding-dino-tiny`. It downloads the locked commit `a2bb814d…` and aborts on any sha256 mismatch with `configs/checkpoints.lock.json`.
   4. GPU smoke test: `python scripts/run_detection.py --config configs/experiments/e0_headroom.yaml --n-images 5`. The device comes from the config (`cuda`).
      - Every `<condition>.jsonl` record must have `status: ok`, and `log.jsonl` must show no errors.
      - Smoke outputs are a pipeline check, not results. Do not commit them.
   5. Full E0: `python scripts/run_detection.py --config configs/experiments/e0_headroom.yaml`. It prints the run directory `outputs/predictions/<run_id>`, with run_id = `e0_headroom-<config hash>`, which differs from the smoke run.
      - Re-running the same command, or passing `--resume outputs/predictions/<run_id>`, continues an interrupted run.
      - Failed images are logged and stay in the denominator.
   6. `python scripts/evaluate_run.py outputs/predictions/<run_id>`. It writes `outputs/metrics/<run_id>/{metrics.json, run_config.json}`, prints the summary table and the bootstrap CIs, and aborts if fast AP50 ≠ COCOeval.
   7. Commit `outputs/metrics/<run_id>/` (predictions are gitignored) and write the E0 report. Then run the context-keeper (step 6).
6. Follow the rules in §3. After finishing a phase, run the **context-keeper** agent (`.claude/agents/context-keeper.md`). It updates this file from repo facts only (§4–§8, §13, §14) and does not commit; the lead reviews and commits.

## 13. Change log
| Date | Phase | Change |
|---|---|---|
| 2026-10-20 | M0/M1 | Initial audit of repo and hardware; literature search; plan approved; git init; directory layout created; CONTEXT.md created |
| 2026-10-20 | M0+M1 end (after 2767b7f) | Context-keeper update: GPU corrected to RTX 3050 Ti; venv, versions, disk, and checkpoint commits recorded (§4); CastDet/OpenRSD/SAS/RT-OVAD verified + citation sweep (§5); H0 falsified, 251/256 (§6); module statuses (§7); D10–D12 added (§8); E0 full vocabulary = single prompt (§10); risks updated (§11); context-keeper agent noted (§9, §12); §14 phase log added |
| 2026-10-20 | M1 | GitHub remote `origin` added and `main` pushed (§4, §9, §11 risk 7, §14) |
| 2026-10-21 | M1 complete (after 2e26ba2) | Context-keeper update: §4 experiment-machine column filled from `outputs/env/LAPTOP-5CPRTI27.json` (RTX 5060 Laptop GPU, cc 12.0, torch 2.7.1+cu128, Python 3.10.11); laptop column refreshed after the version alignment (numpy 2.2.6 / pandas 2.3.3 / matplotlib 3.10.9; C: 8.9 GB; lock 51 packages); git identities corrected; M1 ✅ (§7); D13 reference environment (§8); risks 5–6 updated and 10–11 added (§11); §12 step 3; §14 rows; README status |
| 2026-10-21 | repo | History rewritten at the user's request: Claude co-author trailers removed from all commit messages (code unchanged), then force-pushed. Old to new IDs: 1256354→2767b7f, 5991c38→2b141fc, 1bf7a57→654efbb, 36a5d2b→2e26ba2, d510064→caa24df. All references in this file, the reports, and the env records were updated. **Policy: no Claude attribution in commits; push only as the user.** |
| 2026-10-24 | M2 end (after 2c36846) | Context-keeper update from `reports/M2_dataset.md` + `data/manifests/*`: header; §4 data policy (dev 730 images, 399 MB; RTX 5060 full fetch pending); §5 LAE-80C image count resolved (3,592) and new §5a dataset facts; §6 H0 re-check 249/256 and new H7; §7 M2 ✅ (RTX 5060 full image validation pending), M3 + M9 next, M13 23 tests; §8 D14–D17 added, D18 (D2 executed: dev 730 / test 2,862, seed 2026, ID-list sha256), D19 (D3 refinement: VisDrone mapping deferred to M11); §9 new files; §10 E0 adds the source-label-space oracle and both evaluations; §11 risk 3 resolved, risks 2/4/6/11 updated, risks 12–15 added; §12 data step for the RTX 5060; §14 M2 row |
| 2026-10-24 | M2 close-out | RTX 5060 (Amit, commits 6b70fe3, 89b937b, b95a7be): all 3,592 images OK, zip sha256 identical; env re-audited in the reference venv (Python 3.10.11, cu128, all pins match; nvidia-smi 1665 MB used). Fixes: the 5060 lock file converted from UTF-16 to UTF-8; `env_audit.py --write-lock` added (no shell redirection); `fetch_lae80c_images.py` keeps earlier zip facts on re-run, and the 5060 check file's zip fields restored from 6b70fe3. Open: pytest in the reference env |
| 2026-10-24 | M3 pipeline + M9 evaluator (after 1bc146a) | Context-keeper update from `reports/M3_baseline.md`, the lock file, the E0 config and the new code. Header: next = E0 on the RTX 5060. §4: GDINO-T downloaded and locked (`a2bb814d…`, `model.safetensors` `1a2412ef…`); tests 39; disk / nvidia-smi / audit commits refreshed from the 2026-10-24 env JSONs; mangled venv path fixed (also in §11 risk 11). §6: H0 re-check with the official caption format (249/251 unchanged). §7: M3 🟡, M9 🟡, M13 39 tests, M3+M9 to-do updated. §8: D20 (supersedes D5), D21, D22. §9: new files. §10: E0 config (fp32, not fp16), conditions, metrics. §11: risks 6/10/11/13 updated, risks 16–17 added. §12: E0 runbook (step 5). §14: M2 close-out commit filled in, M3 row |

## 14. Phase log
| Phase | Commit | Key commands | Outcome |
|---|---|---|---|
| M0 literature audit | `2767b7f` | Firecrawl web searches; Semantic Scholar citation sweep (2506.00333, 2601.22164); `python scripts/check_prompt_length.py --classes configs/lae80c_classes_raw.txt` | 13-work matrix; gap holds as of 2026-10-20; H0 falsified (251/256 tokens); gated decision recorded |
| M1 repo & compute audit | `2767b7f` | `python -m venv D:/aerial-ovd-store/venv`; pip install (torch 2.7.1 CPU wheels, transformers 4.57.3, …); `pip install -e .`; `pytest`; `python scripts/env_audit.py`; `python scripts/fetch_checkpoints.py --dry-run` | Laptop env recorded (RTX 3050 Ti, CPU torch); 16/16 tests pass; checkpoint commits resolved (1.77 GB, not downloaded). Pending: RTX 5060 audit, GitHub remote |
| Context-keeper update + fixes | `2b141fc` | context-keeper agent; M0 exit-criteria table added; README fixed; env audit re-run at 2767b7f | M0 ✅; `.claude/agents/context-keeper.md` and `.gitattributes` committed |
| GitHub remote | `654efbb` | `git remote add origin https://github.com/Himanshu-RG/Vision-Aerial-Imagery.git`; `git push -u origin main` | Repo on GitHub; RTX 5060 machine can now clone it |
| M1 complete (RTX 5060 audit) | `2e26ba2` | on `LAPTOP-5CPRTI27` at 654efbb (clean tree): exact commands not recorded in the report; artifacts are `outputs/env/LAPTOP-5CPRTI27.json` (from `scripts/env_audit.py`) and `requirements-lock-LAPTOP-5CPRTI27.txt`; `pytest` run reported by the user (result not recorded) | RTX 5060 env recorded (cc 12.0, CUDA 12.8 available, cuDNN 90701); version drift vs. laptop found (numpy/pandas/matplotlib); GPU-memory anomaly noted |
| M1 version alignment | `caa24df` | exact pins in `requirements.txt`; laptop reinstalled to match; `pytest` (16/16); `python scripts/env_audit.py`; laptop lock refreshed | Both machines on numpy 2.2.6 / pandas 2.3.3 / matplotlib 3.10.9; D13 recorded; M1 ✅; next M2 |
| M2 dataset & frozen protocol (dev laptop `LAPTOP-8N8B1VIP`, 2026-10-24) | `427b885` | `python scripts/prepare_lae80c.py` (2nd run: byte-identical outputs, verified); `python scripts/check_prompt_length.py --classes configs/lae80c_classes_light.txt`; `python scripts/fetch_lae80c_images.py --split dev --delete-zip`; `pytest` | Image count resolved (3,592 images / 86,558 boxes → 86,539 after cleaning, D17); disjoint source label spaces found (H7, D15); split frozen dev 730 / test 2,862 (D18); "light" names 249/256 tokens (D14); dev images 730/730 OK; **23 passed**. M2 ✅ |
| M2 close-out (RTX 5060 data + env fixes) | `1bc146a` (RTX 5060 artifacts: `6b70fe3`, `89b937b`, `b95a7be`) | on `LAPTOP-5CPRTI27`: `python scripts/fetch_lae80c_images.py --split all --delete-zip` (run twice), `python scripts/env_audit.py`; on the laptop: lock re-encoding, `env_audit.py --write-lock`, `pytest` (23 passed) | All 3,592 images OK on the experiment machine; lock files readable; M2 fully closed |
| M3 pipeline + M9 evaluator validation (dev laptop `LAPTOP-8N8B1VIP`, 2026-10-24) | (this commit) | `scripts/fetch_checkpoints.py` (GDINO-T only; lock written 2026-10-24 10:17 UTC); `python scripts/prepare_lae80c.py` (re-run for the official prompt format: only `full_prompt_light_names` changed, split unchanged); `scripts/check_prompt_length.py` (light 249, raw 251, unchanged); `python scripts/run_detection.py --config configs/experiments/e0_headroom.yaml --device cpu --n-images 4` (smoke run `e0_headroom-6690095f1631`: 3 images because of a sampler rounding bug, since fixed); `scripts/evaluate_run.py` on the smoke run (metrics deleted, not results); `pytest` (**39 passed**) | Detector wrapper, E0 conditions, runner, and evaluator built. Known-answer tests and the bootstrap validated; the smoke run's consistency check passed. D20–D22 recorded. M3 🟡 / M9 🟡. **No experimental results yet**; next is E0 on the RTX 5060 |
