# M2 — Dataset Selection, Validation & Frozen Protocol (2026-10-24)

## Objective
Validate LAE-80C, build a manifest, fix the vocabulary and naming policy, freeze the dev/test split, and define the evaluation protocol. All of this happens **before any detector is run**.

## Dataset choice
- **Primary:** LAE-80C (decision D1).
  - Source: HF `jaychempan/LAE-1M` at commit **`60ce78a91a09ea9975eba6de18ffe621936dbc0f`**. MIT license per the dataset card.
  - Raw annotation JSON sha256: `7e848c53…99fb2`.
- **Generalization (M11):** per-source breakdown inside LAE-80C, plus VisDrone-DET as an external set (D3). The VisDrone class mapping is **deferred to M11**: it does not affect the LAE-80C protocol, and the decision is recorded there.

## Files
| Path | Content |
|---|---|
| `configs/data_lae80c.yaml` | Pinned revision, cleaning switches, split fraction/seed/method, COCO size bins |
| `src/aerial_ovd/data/lae80c.py` | `load_coco`, `source_of`, `source_label_space`, `clean_coco` (logged cleaning), `build_manifest`, `subset_coco` |
| `src/aerial_ovd/data/split.py` | Deterministic multi-label iterative stratification (Sechidis et al., 2011) |
| `src/aerial_ovd/vocabulary/names.py` | Naming policy (`light`, `raw`) and prompt builder |
| `scripts/prepare_lae80c.py` | Clean → manifest → frozen split → vocabulary → report. Refuses to overwrite a frozen split |
| `scripts/fetch_lae80c_images.py` | Download the pinned image zip, extract a split, validate every image |
| `tests/test_data_lae80c.py` | 7 tests (cleaning log, no input mutation, manifest, label space, split determinism/disjointness/stratification, naming) |
| `data/manifests/lae80c_manifest.csv` | 3,592 rows: id, file, source, split, size, n_boxes, n_classes, classes, n_small/medium/large, n_tiny_flag |
| `data/manifests/lae80c_{dev,test}_ids.txt` | **Frozen split** (dev 730 / test 2,862) |
| `data/manifests/lae80c_vocabulary.csv` | 80 rows: id, raw name, prompt name, source, n_boxes, n_images_dev/test |
| `data/manifests/lae80c_prepare_report.json` | Cleaning log, statistics, sha256 of inputs and outputs |
| `<store>/datasets/LAE-80C-clean/{all,dev,test}.json` | Cleaned COCO ground truth (not committed; sha256 in the report) |

## Commands
```bash
python scripts/prepare_lae80c.py                                # 2nd run: byte-identical outputs (verified)
python scripts/check_prompt_length.py --classes configs/lae80c_classes_light.txt  # 249 / 256 tokens
python scripts/fetch_lae80c_images.py --split dev --delete-zip  # dev laptop
python scripts/fetch_lae80c_images.py --split all --delete-zip  # RTX 5060 (to do)
pytest                                                          # 23 passed
```

## Findings (actual)
1. **Image count resolved:** 3,592 images and 86,558 boxes, matching the paper that cites 2601.22164. The HF README's "20k (val)" does not describe this file.
2. **The five sources have disjoint label spaces.** Every image is annotated **only** for its own source's classes:

   | Source | Images | Classes | Boxes/img | Small boxes | Classes/img |
   |---|---|---|---|---|---|
   | FAIR-18C | 2,130 | 18 (ids 25–42) | 18.0 | 91% | 2.50 |
   | Xview-34C | 758 | 34 (ids 47–80) | 59.9 | 34% | 2.29 |
   | DIOR-18C | 536 | 18 (ids 1–18) | 4.0 | 30% | 1.43 |
   | DOTA-6C | 137 | 6 (ids 19–24) | 3.8 | 17% | 1.07 |
   | PP-4C | 31 | 4 (ids 43–46) | 4.8 | 3% | 2.71 |

   Several concepts appear under different names in different sources, e.g. `vehicle` (DIOR), `Small Car` (FAIR) and `Passenger Car` (xView), or `airplane` (DIOR) and `Fixed-wing Aircraft` / `Small Aircraft` (xView). Each name is annotated only on its own source's images. Under standard COCO evaluation, a correct detection under another source's name counts as a false positive.
3. **Image size identifies the source:** 600² is always FAIR, 800² always DIOR, 1024² is xView or DOTA, and irregular sizes (~1052–1259 × ~683–754) are PP. This is a shortcut that selectors must never see (rule D16).
4. **Box format:** horizontal boxes only. All 45,375 xView polygons are axis-aligned rectangles that match their bbox within 1 px. Any conversion from oriented boxes for the DOTA/FAIR sources was done **upstream by the LAE-1M authors**; we perform none.
5. **Cleaning (`lae80c_prepare_report.json`):**
   - **972 boxes clipped** to image bounds. All are FAIR or DOTA objects cut at tile edges; overflow median 4 px (FAIR) / 9.5 px (DOTA); no box lies fully outside; at least 58.4% of each box's area is kept.
   - **19 exact duplicates removed** (all FAIR).
   - 0 non-positive boxes, 0 bad category ids, 0 duplicate ids or file names, 0 images without annotations.
   - **23 boxes with a side < 2 px** (21 xView, 2 FAIR) are kept and flagged.
   - After cleaning: 86,539 boxes.
6. **Statistics:**
   - Size share: 59.3% small, 35.1% medium, 5.7% large.
   - Boxes per image: mean 24.1, max 668. Classes per image: mean 2.24. The prior paper's "3.2" is not reproduced from this file, and the reason is unknown.
7. **Long tail:** `helipad` occurs in **1 image** in the whole benchmark, which is in test. `container crane`, `Flat Car`, `Straddle Carrier` and `Reach Stacker` occur in 3–5 images each. Per-class AP for these is not meaningful. Rare-class analysis will pool classes by frequency band, not report single-class numbers.

## Frozen split (D2)
- **Method:** iterative stratification over class presence, seed 2026, dev fraction 0.20.
- **Result:** dev 730 / test 2,862.
  - DIOR 108 / 428, DOTA 28 / 109, FAIR 435 / 1,695, PP 8 / 23, xView 151 / 607.
  - Dev share of small boxes: 62.3% (test: 58.5%). Classes per image: dev 2.21, test 2.25.
- **Missing from dev:** 1 class (`helipad`). 16 classes have only 1–2 dev images.
- ID-list sha256:
  - dev `131348ae…7f8c`
  - test `1d6b2c39…cd0e`
- `prepare_lae80c.py` refuses to write a different split unless `--allow-resplit` is passed, and that requires a new decision row.

## Vocabulary & naming policy (D14)
- **Master vocabulary:** all 80 LAE-80C categories. Evaluation always uses category ids.
- **Policy "light":**
  - lower-case everything;
  - split 7 glued DIOR names (`groundtrackfield` → `ground track field`, also baseballfield, basketballcourt, storagetank, tenniscourt, trainstation, golffield);
  - nothing else: no synonyms, no prefixes, slashes kept, overlapping classes kept as defined.
- The same names are used by every method.
- **Token check:** the full prompt is **249 / 256** tokens, so it is still a single prompt (D10).

## Evaluation protocol (D15) — implemented in M9, frozen now
- **Primary: source-aware ("federated") COCO evaluation.** For each image, only predictions whose class belongs to the image's **source label space** are scored. Predictions of other classes are ignored (neither TP nor FP), because those classes were never annotated on that image.
  - Source identity is used **only by the evaluator**, never at inference.
  - **Assumption:** each source annotates its own classes exhaustively. This is standard for DIOR/DOTA/FAIR1M/xView but not verified by us.
- **Secondary: standard all-80-class COCO evaluation**, reported for comparability with arXiv 2601.22164.
- **Metrics:**
  - mAP@50 (headline), mAP@50:95, AR@100;
  - per-class AP, with rare classes pooled by frequency band;
  - P / R / F1 at a score threshold chosen on dev.
- **Oracle diagnostics (never methods):**
  - (i) **per-image GT vocabulary**;
  - (ii) **source-label-space vocabulary** (the 6–34 classes of the image's source).

  Comparing (i) and (ii) splits the oracle headroom into a "which label space applies" part and a "which classes are in this image" part.
- **Open-vocabulary caveat:** no local training happens, so every class is "unseen" locally. But Grounding DINO's pretraining (O365, GoldG, Cap4M for Swin-T) certainly covers concepts such as car, airplane and ship. **We make no novel-class claims.** If a "novel" breakdown is needed, it would require auditing overlap with the pretraining vocabulary.

## New hypothesis
**H7:** a substantial share of the full-vs-oracle gap under standard evaluation comes from cross-source label-space mismatch.
- **Prediction:** the gap shrinks under source-aware evaluation, and the source-label-space oracle recovers much of the per-image-oracle gain.
- **Falsified if** the gap under source-aware evaluation stays within the bootstrap CI of the standard-evaluation gap.

## Selector input rule (D16)
Vocabulary selectors receive only the decoded image pixels, resized/cropped by the selector's own fixed preprocessing.
- **Forbidden inputs:** file name, source tag, original image size, EXIF/metadata, annotation-derived information.
- Unit tests in M4 will check that the selector API does not accept these.

## Image validation
`scripts/fetch_lae80c_images.py` records missing, unreadable, and size-mismatched images per split and host in `data/manifests/lae80c_images_check_<host>_<split>.json`.
- **Dev laptop, dev split: 730 / 730 images OK.** 0 missing, 0 unreadable, 0 size mismatches.
  - `images.zip` sha256: `288988fe25a3d7173c5085d0660b785bdc6a819559db3d7af78e60a1ee7cb398`.
  - The zip has 3,592 members, all listed in the annotations, with no extras.
  - The zip was deleted after extraction. The dev images take 399 MB at `<store>/datasets/LAE-80C-images`.
- **RTX 5060, all images: 3,592 / 3,592 OK** (commits 6b70fe3 and b95a7be by Amit). The zip sha256 is identical to the laptop's (`288988fe…`).
  - The second run found the images already extracted and dropped the zip fields from the check file. They were restored from 6b70fe3, and `fetch_lae80c_images.py` now keeps them on re-runs.

## Known limitations
- Exhaustive within-source annotation is assumed, not verified.
- The class-per-image discrepancy with 2601.22164 (2.24 vs. 3.2) is unexplained.
- Single-image classes make per-class AP unreliable.
- `PP-4C` has only 31 images (8 in dev).

## Exit criteria
| Criterion | Status |
|---|---|
| Validated dataset manifest | ✅ annotations; ✅ dev images (laptop, 730/730); ✅ all images (RTX 5060, 3,592/3,592) |
| Documented vocabulary + naming policy | ✅ |
| Frozen evaluation protocol | ✅ defined (D15); implemented and validated in M9 |
| Frozen split committed before any detector run | ✅ |

## Next: M3 (Grounding DINO baseline) + M9 (metric validation)
1. Fetch the GDINO-T checkpoint (pinned; sha256 lock).
2. Implement detector inference with positive-map class scoring (D5).
3. Implement the source-aware and standard COCO evaluators, with known-answer tests (GT as predictions → AP = 1.0).
4. CPU smoke test on a few dev images on the laptop.
5. E0 headroom check on the RTX 5060, which now also includes the source-label-space oracle.
