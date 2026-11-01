# M11 — Generalization (part 1: across LAE-80C sources, dev; 2026-11-01)

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

## Part 2: external dataset NWPU VHR-10 (D45; step 1 done, detection pending)
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

### Detection (pending)
Config: `configs/experiments/m11_nwpu_detection.yaml`, with 9 conditions: full, oracle, random, frequency top-8, OpenAI CLIP top-8/20, RemoteCLIP top-3/8 and headline. The CPU smoke run is OK.
