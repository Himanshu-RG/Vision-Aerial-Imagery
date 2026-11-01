# M11 — Generalization (part 1: across LAE-80C sources, dev, 2026-11-01; part 2: external NWPU VHR-10, 2026-11-02)

## Method
- **Data:** per-source AP50 from committed metrics (`scripts/m11_per_source.py`), with no new GPU runs.
  - Under source-aware scoring, a class's ground truth and kept predictions lie only in its own source's images.
  - So the mean per-class AP50 over a source's classes equals that source's AP50.
- **Runs:** E5 run (headline, RemoteCLIP, caption, TagCLIP-style) and M4 run (OpenAI CLIP control), the same 200 dev images.
- **Tables:** `reports/generated/M11_per_source_e5_caption_tagclip_det.md` and `..._m4_selectors_det.md`.
- **Statistics:** per-class sign test vs full. Source groups with few images (DOTA 8, PP 2) are indicative only.

## RemoteCLIP's training data (verified from the paper)
From Liu et al., arXiv 2306.11029 v4 (IEEE TGRS), Table I:

| | datasets |
|---|---|
| Detection group "DET-10" | AUAIR, CARPK, **DIOR**, **DOTA**, HRRSD, HRSC, LEVIR, RSOD, Stanford, **VisDrone** (the text names only six; the table lists ten) |
| Segmentation group | iSAID, LoveDA, Potsdam, Vaihingen |
| Retrieval group | RSICD, RSITMD, UCM |

- **xView and FAIR1M are not included.**
- The splits used are not stated. LAE-80C uses DIOR/DOTA validation images, so the overlap is at least at the domain and class-name level. Image-level overlap cannot be ruled out from the paper.
- This makes LAE-80C's sources a natural split: **seen by the selector encoder** (DIOR, DOTA) vs **unseen** (FAIR1M, xView, PP).

## Results (source-aware AP50, 200 dev images)
| condition | seen: DIOR + DOTA (22 classes) | unseen: FAIR + xView + PP (53 classes) |
|---|---|---|
| full | 0.069 | 0.011 |
| oracle | 0.262 (20/2, p = 1e-4) | 0.090 (46/1, p = 7e-13) |
| RemoteCLIP top-8 | 0.177 (19/0, p = 4e-6) | 0.009 (16/16, p = 1) |
| **headline (D34)** | **0.223** (18/3, p = 0.002) | **0.007** (17/15, p = 0.86) |
| OpenAI CLIP top-8 (control) | 0.128 (17/0, p = 2e-5) | 0.012 (14/13, p = 1) |
| OpenAI CLIP top-20 (control) | 0.116 (15/2, p = 0.002) | 0.022 (24/9, p = 0.014) |

In brackets: classes better / worse than full, per-class sign test.

Per source (headline vs full): DIOR 0.060 → 0.220, DOTA 0.113 → 0.240, FAIR 0.011 → 0.011, xView 0.013 → 0.005, PP 0 → 0.009 (2 images).

## Findings
1. **The headline gain does not generalize to sources unseen by its encoder.**
   - All of it is on DIOR and DOTA (×3.2).
   - On FAIR1M and xView it is zero or negative, even though the oracle shows ×8 headroom there.
   - The headline result must be stated with this qualification.
2. **It is not only exposure.**
   - OpenAI CLIP, which was never trained on DIOR or DOTA, also gains on them (×1.7–1.9). Those datasets have coarse, visually distinctive classes such as stadium, airport and windmill, which any CLIP can rank.
   - RemoteCLIP gains more on them (×2.6 at top-8, ×3.2 for the headline), which is consistent with an exposure advantage.
3. **On the unseen sources, natural-image CLIP does better than RemoteCLIP.** OpenAI CLIP top-20 is the only selector with a significant gain there (0.011 → 0.022). RemoteCLIP at top-20 gives 0.011 (22/12, p = 0.12).
4. Combined with M10, failure on the unseen sources has two causes: fine-grained, dense vehicle and ship classes, and cross-source synonyms. A better aerial encoder is not enough there; the encoder needs to be trained on those classes.

## Pre-declared for the one-time test (D44, before any test evaluation)
- Every test result is reported for the whole test split and **separately for the seen (DIOR+DOTA) and unseen (FAIR+xView+PP) groups**, plus per source, under both protocols.
- The test conditions are listed in `configs/experiments/final_test.yaml`, to be created and committed before the test run.

## Part 2: external dataset NWPU VHR-10 (D45; done 2026-11-02)
**Choice.** VisDrone, the external set planned in D3, is in RemoteCLIP's training data, so the user chose NWPU VHR-10 instead (2026-11-01). It is not in RemoteCLIP's training data, except for its colour-infrared images, which are cropped from Vaihingen (RemoteCLIP's segmentation group). Its classes are DIOR's names, since the same group built both datasets. So it tests **unseen images with familiar classes**.

**Data.** `scripts/prepare_nwpu.py`, HF mirror `SGhazalS/nwpu-vhr10` pinned at revision `f7aa667`. The dataset is research use only; images stay in the local store and are not committed.
- 650 positive images → 614 kept, with 3,594 boxes in 9 classes, mapped to the DIOR ids of the 80-class vocabulary.
- Ship (302 boxes) has no single equivalent and is dropped. That leaves 36 ship-only images with nothing to evaluate.
- The per-class counts match the published dataset.
- **Colour-infrared subgroup:** images 421–453 (33), found by a colour rule plus a visual check (D45a).

**Protocol.**
- The same 80-class prompt is used, and every method is applied **frozen**, with no recalibration on NWPU.
- Standard COCO evaluation over the 9 classes. Source-aware scoring is identical here, because other classes have no ground truth.

### Selection quality (detector-free, 614 images)
Source: `reports/generated/M11_nwpu_selection.md`.

| rule | mean size | object-weighted coverage | colour-infrared images | others |
|---|---|---|---|---|
| headline (D34, frozen) | 3.9 | 0.777 | 1.000 | 0.749 |
| RemoteCLIP top-8 | 8 | 0.828 | 1.000 | 0.807 |
| OpenAI CLIP top-8 | 8 | 0.743 | 0.369 | 0.789 |
| OpenAI CLIP top-20 | 20 | 0.920 | 0.872 | 0.926 |
| frequency prior top-8 | 8 | 0.000 | 0 | 0 |

- **The frozen threshold transfers, but its size does not.** It keeps 3.9 classes per image on NWPU against 8 on LAE-80C: RemoteCLIP is more confident on these DIOR-like images.
- **Exposure is visible.** On the Vaihingen-derived images, RemoteCLIP covers 100% of objects and OpenAI CLIP 37%. On the rest they are close (0.81 vs 0.79 at top-8).
- **Near-synonyms are selected again:** Engineering Vehicle and Passenger Car for "vehicle", Small Aircraft and Fixed-wing Aircraft for "airplane". This is the M10 finding on a new dataset.
- The frequency prior learned on LAE-80C dev never selects an NWPU class: it carries over no information.

### Detection (614 images, RTX 5060)
Run `m11_nwpu_det-58d87bbcbac5` (Amit, commit `3d023f2`):
- Code `26de481`, clean; scored with the empty-set fix `5a23870`.
- 9 conditions, 0 failures, 3,594 ground-truth boxes.
- The GPU was not idle at the start (1.9 GB used, 38% utilisation), so the latencies are indicative only.

Sources: `reports/generated/m11_nwpu_det_tables.md` (vs full), `m11_nwpu_det_tables_vs_headline.md` (vs the headline) and `M11_nwpu_subgroups.md`. AP50@900; standard and source-aware are identical here.

| condition | classes / image | AP50 | Δ vs full (robust?) | colour-infrared (33) | other (581) | F1@0.25 | latency ms |
|---|---|---|---|---|---|---|---|
| full | 80 | 0.088 | ref | 0.000 | 0.088 | 0.082 | 668 |
| oracle_gt (diagnostic) | 1.4 | 0.208 | +0.120 (yes) | 0.363 | 0.220 | 0.311 | 537 |
| random_matched (control) | 1.4 | 0.007 | −0.081 | 0.000 | 0.007 | 0.015 | 540 |
| frequency prior top-8 | 8 | 0.000 | −0.088 (yes) | 0.000 | 0.000 | 0.000 | 549 |
| OpenAI CLIP top-8 | 8 | 0.101 | +0.013 (no: sign test 7/2, p = 0.18) | 0.020 | 0.102 | 0.108 | 546 |
| OpenAI CLIP top-20 | 20 | 0.100 | +0.012 (yes) | 0.011 | 0.100 | 0.090 | 567 |
| RemoteCLIP top-3 | 3 | 0.074 | −0.014 (no) | 0.334 | 0.070 | 0.083 | 541 |
| RemoteCLIP top-8 | 8 | 0.093 | +0.005 (no) | 0.121 | 0.093 | 0.134 | 546 |
| **headline (D34, frozen)** | 3.9 | **0.129** | **+0.041 [+0.030, +0.057] (yes; 8/1 classes)** | 0.311 | 0.131 | **0.183** | 543 |

Per class (AP50): source `M11_nwpu_subgroups.md`.

| class | full | headline | oracle |
|---|---|---|---|
| airplane | 0.742 | 0.672 | 0.922 |
| ground track field | 0.007 | 0.243 | 0.250 |
| vehicle | 0.000 | 0.163 | 0.388 |
| harbor | 0.000 | 0.022 | 0.171 |
| baseball field | 0.004 | 0.021 | 0.053 |
| bridge | 0.027 | 0.028 | 0.032 |
| basketball court | 0.012 | 0.013 | 0.019 |
| storage tank | 0.000 | 0.000 | 0.037 |
| tennis court | 0.000 | 0.000 | 0.002 |

### Findings (part 2)
1. **The frozen headline method transfers to a dataset its encoder was not trained on.**
   - AP50 rises from 0.088 to 0.129 (×1.47), robust on all three D26 checks, with no recalibration.
   - That recovers 34% of the oracle headroom (0.041 of 0.120). On the seen DIOR+DOTA sources of LAE-80C dev it recovered 80% (0.154 of 0.193).
   - The gain does **not** come from the colour-infrared images RemoteCLIP saw: on the other 581 images it is still 0.088 → 0.131.
   - Precision–recall also improves: F1@0.25 0.082 → 0.183; best F1 0.087 → 0.184 (diagnostic).
2. **It beats every other frozen selector on NWPU.**
   - Against the headline (`_vs_headline` table), OpenAI CLIP top-8 is robustly worse (−0.029).
   - RemoteCLIP top-3/top-8 and OpenAI CLIP top-20 are also worse on the point estimate (−0.036 to −0.056; CIs exclude 0), but their per-class sign tests are not significant (9 classes only).
   - **Fixed-size RemoteCLIP top-k gives no gain** (+0.005 at k = 8, −0.014 at k = 3), while the threshold rule at a similar average size (3.9) does. Here the benefit comes from the **adaptive size**, not from the encoder alone: 46% of NWPU images keep only 1–2 classes (10% on LAE-80C dev), and the rest keep up to 17 (median 3).
3. **The gain is concentrated in two classes.**
   - Ground track field (0.007 → 0.243, almost the oracle) and vehicle (0.000 → 0.163). With all 80 classes, their scores are probably diluted by related LAE-80C names (e.g. stadium; the many FAIR/xView vehicle types, cf. M10). This is an interpretation; the confusions were not counted on NWPU.
   - Airplane, the class the full vocabulary already detects, loses 0.07.
   - Storage tank and tennis court stay at 0 even with the oracle: a detector limitation, not a selection one.
4. **Encoder exposure is visible again.**
   - On the Vaihingen-derived colour-infrared images, the full vocabulary scores 0 and OpenAI CLIP selectors ≤ 0.02, while every RemoteCLIP-based selector scores 0.12–0.33 (oracle 0.36).
   - This matches selection coverage (1.00 vs 0.37) and part 1.
5. **The frequency prior learned on LAE-80C fails completely** (AP 0): none of its 8 classes is an NWPU class. This is what a dataset-specific prior does under a label-space shift, and it is the only condition whose source-aware prediction set is empty.
6. **Latency:** the headline's median is 543 ms against 668 ms for the full vocabulary (−19%, as on dev). The GPU was not idle, so M12 re-measures this.

### What this means for the claim (read with part 1)
- NWPU VHR-10 is not in RemoteCLIP's listed training data, and the frozen method still gains there.
- But NWPU was built by the same group as DIOR, from the same imagery source (Google Earth, plus Vaihingen crops), and all its classes are DIOR classes. So it tests **new images from a familiar domain and taxonomy**.
- Combined with part 1 (no gain on FAIR1M/xView, whose fine-grained classes are unfamiliar to the encoder), the supported claim is:
  > The selector's gain transfers to new images and new datasets whose classes and imagery resemble what its encoder was trained on, but not to fine-grained, unfamiliar label spaces.
- **Limitation:** image overlap between NWPU VHR-10 and DIOR (whose training images RemoteCLIP used) was not checked. DIOR's training images are not available on our machines. Overlap would make the NWPU gain partly an exposure effect.
- **Analysis choices:** the subgroup and per-class numbers are point estimates from `scripts/m11_nwpu_subgroups.py`; only the overall comparisons carry the D26 checks.
