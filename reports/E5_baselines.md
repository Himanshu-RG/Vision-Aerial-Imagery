# E5 — Caption-based (VocAda-style) and TagCLIP-style baselines (step 1 done, 2026-10-31; detection pending)

Both are **labelled baselines, not reproductions** (D38).

## Caption-based (VocAda-style)
1. **Caption:** Florence-2-base (`florence-community/Florence-2-base`, sha256-locked), task `<MORE_DETAILED_CAPTION>`, beams 3, ≤ 128 new tokens, no sampling. VocAda used LLaVA-Next-7B, which does not fit the 8 GB GPU.
2. **Phrases:** all 1–3-grams of consecutive non-stopword tokens. VocAda used spaCy noun phrases; we avoided a new dependency on both machines.
3. **Class score:** max over phrases of the OpenAI CLIP ViT-B/32 text–text cosine. VocAda used CLIP ViT-L/14 text similarity.
4. **Selection:** top-k, plus a threshold calibrated on the 530 calibration images to mean size 3 / 8 (D32).

All 730 dev images were captioned with 0 failures and 0 empty captions (mean 36.2 phrases per caption). CPU time was ~5.6 s per image; GPU timing is pending. The captions are committed in `outputs/selection/dev/captions_florence2_base.json`.

## TagCLIP-style (crop-level analogue)
OpenAI CLIP, local-only, max over all 29 crops (M5 variant `a0_max_g234`). TagCLIP itself classifies patch tokens with attention refinement. This keeps only its local-to-global "max over local evidence" idea.

## Selection quality (200 held-out dev images)
Source: `reports/generated/E5_selection_tables.md`.

| k | RemoteCLIP (ours, base) | caption (VocAda-style) | TagCLIP-style |
|---|---|---|---|
| 3 | 0.217 / 0.295 | 0.181 / 0.115 | 0.174 / 0.223 |
| 8 | 0.388 / 0.508 | 0.355 / 0.212 | 0.338 / 0.375 |
| 20 | 0.662 / 0.697 | 0.533 / 0.411 | 0.630 / 0.631 |

Each cell is image vocabulary recall / macro (class-averaged) pair recall. Caption − RemoteCLIP image recall: −0.037 (k=3, n.s.), −0.033 (k=8, n.s.), **−0.129 (k=20, CI [−0.202, −0.053])**.

### Findings
- **The caption selector is comparable on image-level recall at small k but much worse on class-averaged recall**, less than half of RemoteCLIP's at k = 3/8. It repeatedly proposes the same generic classes, and a natural-image captioner rarely names aerial-specific categories (e.g. windmill and train-station images are described as "desert landscape" and "city"). Class-averaged recall predicted detection AP in M4–M6 (D27), so a weaker detection result is expected.
- **TagCLIP-style is below RemoteCLIP at every k.** Natural-image CLIP on crops is weak (cf. M5).
- **Cost:** captioning is ~5.6 s/image on CPU, far above the ~85 ms latency budget (VocAda reports 5–11 s/image on a V100).

## Detection (pending)
Config: `configs/experiments/e5_caption_tagclip_detection.yaml`. 11 conditions; CPU smoke run OK.
