# M10 — Failure analysis (dev only, 2026-11-01)

## Run and method
- **Predictions:** E5 run `e5_caption_tagclip_det-69e357c3b1f4` (code `222905e`, RTX 5060), the same 200 dev images as E0–M9. The analysis ran on Amit's laptop; results were pushed as `1365fef`.
- **Conditions:** full, oracle, RemoteCLIP top-8, headline (threshold m≈8), caption threshold m≈8.
- **Definitions (fixed before the run, D40):** `src/aerial_ovd/evaluation/failure.py`.
  - **Per ground-truth object:** one of `vocab_miss`, `detected` (coverage at IoU 0.5), `wrong_class`, `poor_box` (IoU 0.1–0.5), `missed`.
  - **Per detection with score ≥ 0.14 (D35):** one of `tp`, `duplicate`, `wrong_class`, `poor_box`, `bg_present`, `bg_absent`, plus the share whose class is never annotated in that image's source.
- **Strata:** object size (tiny < 16² px), objects per image, class rarity (terciles of dev images per class), source.
- **Tables:**
  - `reports/generated/M10_failure_tables.md`
  - `reports/generated/M10_vocab_coverage.md`, from `scripts/m10_vocab_coverage.py`, detector-free and run here.
- **Checks:**
  - The full and oracle vocabularies have 0 `vocab_miss`.
  - Oracle has 0 `bg_absent` and 0 out-of-source detections.
  - The headline's `vocab_miss` (0.740) equals 1 − its object-weighted coverage (0.260), computed independently from the committed selection masks.

Everything below is **descriptive, on dev, without confidence intervals.** Strata with few images are marked.

## Findings

### 1. Most objects are lost before the detector runs: the vocabulary drops their class
| selector | objects whose class is selected | image-averaged class recall (M4–M6 metric) |
|---|---|---|
| RemoteCLIP top-8 | 0.249 | 0.388 |
| **headline** | **0.260** | 0.412 |
| caption threshold m≈8 | **0.782** | 0.386 |

- **The per-image metric hides this.** Dense FAIR1M/xView scenes hold most objects (4,061 of 4,788 are in images with > 20 objects), but they count as one image each.
- **The headline method keeps the right class for only 22% of FAIR objects and 24% of xView objects.** It keeps 88–100% for DIOR, DOTA and PP.
- **Among the objects whose class *is* kept, detection is close to the oracle.** The headline covers 48% of them at IoU 0.5 (0.126 / 0.260), against 58% for the oracle. Where the vocabulary is right, the detector behaves almost as if it had the perfect vocabulary.
- **This also explains E5's pooled-F1 result.** The caption selector covers 78% of objects because captions name the vehicles and ships that make up dense scenes. It still loses on class-averaged AP, because it rarely names the less common classes.

### 2. The main cause is near-synonym classes from other source datasets
- LAE-80C merges 5 label spaces with near-duplicate names. Examples: FAIR "Passenger Ship" / "Motorboat" vs xView "Ferry" / "Yacht" / "Sailboat"; FAIR "Small Car" vs xView "Passenger Car" vs DIOR "vehicle".
- **59% of the classes the headline method selects are not annotated in the image's source dataset.** On FAIR images it most often picks Ferry (44 of the FAIR images), Yacht (36), Locomotive (27) and DIOR's "vehicle" (25), all xView or DIOR classes.
- **Diagnostic (uses source identity, so it is not a method):** restricting RemoteCLIP top-8 to the image's source label space raises object coverage from **0.25 to 0.68**, and from 0.25 to 0.67 on FAIR.
- **Source-aware evaluation does not fix this.** It removes the wrong-synonym *detections*, but the correct class is still absent from the prompt.
- The confusion lists agree: the most frequent headline losses are Van → Small Car, Small Car → Passenger Car, Van → Passenger Car and Van → Bus. Each is a `vocab_miss` where the detector boxed the object under a selected near-synonym.

### 3. Dense scenes: selection helps sparse images and hurts dense ones
| condition | 1–5 objects (105 images) | 6–20 (42) | 21–100 (45) | > 100 (8, few) |
|---|---|---|---|---|
| full | 0.025 | 0.007 | 0.038 | 0.032 |
| oracle | 0.214 | 0.104 | 0.126 | 0.062 |
| headline | **0.109** | 0.031 | **0.017** | **0.002** |
| caption m≈8 | 0.031 | 0.017 | 0.016 | 0.020 |

Each cell is AP50 (standard, maxDets 900) on that image subset.

- The headline gain comes from sparse images (×4.3 over full).
- **In images with > 20 objects it is below the full vocabulary**, consistent with finding 1.
- Even the oracle loses most of its accuracy in dense scenes. Its `poor_box` share is 0.40 for the > 100 bin, against 0.08 for 1–5. Crowding is also a detector problem.

### 4. Rare classes: limited by the detector, not the selector
| condition | rare (23 classes) | common (25) | frequent (27) |
|---|---|---|---|
| full | 0.0005 | 0.035 | 0.008 |
| oracle | 0.050 | 0.248 | 0.118 |
| headline | 0.0006 | **0.150** | 0.013 |

Each cell is the mean per-class AP50.

- Rare classes are near zero for every vocabulary, and only 3% of rare-class objects are detected at 0.14 even with the oracle. Better selection cannot rescue them.
- **The headline's gains are concentrated in mid-frequency classes** (×4.3 over full).
- **Frequent classes**, mostly the dense vehicle and ship classes, stay far below the oracle (0.013 vs 0.118), again through finding 2.

### 5. Tiny objects: not the worst size bucket under the oracle
With the oracle vocabulary at 0.14, the detected shares are: tiny 0.21, small 0.18, **medium 0.09**, large 0.54.
- Medium objects have the most poorly placed boxes (0.36 at any score). These are mostly xView objects, where the oracle has a 0.37 `poor_box` share.
- This supports D37: small-object weakness is not simply a matter of pixels.

### 6. Similar classes: fine-grained vehicles dominate wrong-class errors
- With the oracle (the class is in the prompt, but the box gets another class): Van → Small Car (232 objects), Dump Truck → Small Car (47), Small Car → Van (23), Dump Truck → Van (13), Engineering Ship → Dry Cargo Ship (4).
- For the headline method, Motorboat → Ferry (12) is the largest wrong-class pair. That is again a cross-source synonym.

### 7. False detections
| condition | confident detections | TP share | background, class absent from image | out-of-source class |
|---|---|---|---|---|
| full | 28,580 | 0.008 | 0.853 | 0.774 |
| headline | 6,747 (−76%) | 0.027 | 0.670 | 0.567 |
| oracle | 3,875 | 0.209 | 0 | 0 |

"Confident" means score ≥ 0.14.

- **Full vocabulary:** 77% of its confident false detections are classes never annotated in that image's source dataset. The D15 label-space artefact dominates its errors.
- **Headline:** it removes three quarters of the confident detections and triples the TP share. Its remaining false detections are still mostly out-of-source synonyms (57%) and wrong classes (21%).

### 8. Caveat: annotations are incomplete
Some "false" detections are unlabelled objects. Examples:
- In `rare_vocab_miss.jpg` (an xView harbour), dozens of boats have no ground truth.
- FAIR1M has no building class, so houses on FAIR images detected as an xView "Building" count as errors.

This inflates false-detection counts for every condition alike. It does not change the comparisons.

## Qualitative examples (`reports/figures/m10/`, chosen by fixed rules, D40)
| file | rule | image |
|---|---|---|
| `tiny_objects.jpg` | most undetected tiny objects (headline) | FAIR 2448, 114 objects |
| `dense_scene.jpg` | most objects | xView 3027, 650 objects |
| `similar_classes.jpg` | most wrong-class outcomes (headline) | FAIR 2154: marina boats |
| `rare_vocab_miss.jpg` | most rare-class objects dropped by the vocabulary | xView 3466: harbour |
| `selection_gain.jpg` | largest TP gain, headline vs full (side by side) | xView 3229: village buildings |
| `success.jpg` | highest detected share (≥ 3 objects) | DIOR 24 |

Colours: green = ground truth, blue = true positive, red = false positive (score ≥ 0.14).

## Implications
1. **The headline result (D34) stands as measured.** M10 explains *where* it wins (sparse scenes, mid-frequency classes, DIOR/DOTA) and *where* it fails (dense FAIR/xView scenes).
2. **The single largest failure of deployable selection on LAE-80C is cross-source synonym classes.** That is partly a property of the merged benchmark taxonomy, and partly a real problem for any open-vocabulary prompt containing near-synonyms.
3. **A deployable fix exists and is not yet tested:** expand each selected class with its near-synonyms by class-name text similarity. That uses no source identity or ground truth. It would be a new method developed after this dev analysis, so it needs a decision (D41, pending with the user). Test data remains untouched either way.
4. **Rare classes, crowded scenes and fine-grained vehicles are detector limits** that remain even with a perfect vocabulary. They are reported as limitations (D37).
