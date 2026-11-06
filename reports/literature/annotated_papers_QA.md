# Literature review: annotated papers, research questions and answers

Team Transmontane: Amit Adhikari, Himanshu Ray, Saurav Kumar Gupta, Avishek Chowdhury.

Project: *Adaptive Vocabulary Selection for Robust Open-Vocabulary Object Detection in Aerial Imagery.*
- Detector: frozen Grounding DINO-T.
- Benchmark: LAE-80C.
- Selector: a RemoteCLIP per-image vocabulary selector with an adaptive threshold.


## **Highlight colours** in the annotated PDFs:

| Colour | Meaning |
|---|---|
| Yellow | A problem we share |
| Green | A related method |
| Blue | A dataset, metric or number |
| Pink | A gap or limitation |



---

## 1. WeDetect: Fast Open-Vocabulary Object Detection as Retrieval (Fu et al., CVPR 2026)


**Q1. Why do fusion-based detectors such as Grounding DINO get slower as the vocabulary grows, and how does WeDetect avoid this?**
- **Why fusion is slow.** The way we understood it, the fusion layers mix the text into the image features, so the features become "query-specific" and can't be reused for a different prompt. Every new set of class names means running the heavy part of the model again.
  - The authors give an example that stuck with us: Grounding DINO on LVIS (1,203 classes) with chunks of 40 classes needs 31 separate forward passes, which they say means "several seconds of latency per image".
- **WeDetect's design.** They simply remove the fusion and go back to a CLIP-like dual-tower setup:
  - text side: an XLM-RoBERTa text encoder;
  - image side: a ConvNeXt backbone, a CSPRepBiFPAN neck and a YOLO-World contrastive head, and no fusion anywhere in the neck.
- **Effect.** Classification is only a dot product between the image grid features and the class text embeddings, so the text embeddings can be computed one time and kept.
- **Speed (Table 1):**
  - WeDetect-Tiny runs at 62.5 FPS, while Grounding DINO-T (172M parameters) runs at 6.0 FPS.
  - Base and Large are 6× and 3× faster than Grounding DINO and still more accurate. This surprised us a little, since we had assumed fusion was needed to get good accuracy.
- *(Our reading)* This is basically why our selector also made things faster. GDINO's cost depends on the prompt, so when we give it fewer classes per image it gets quicker: 414 → 343 ms on the RTX 5060.

**Q2. How was WeDetect's 15M-image training set built, and do the multi-granularity labels matter?**
- **Images.** They sampled images from SAM-1B, LAION, CC12M, Zero and some licensed websites. Where an image had a raw caption, they used it to find rare nouns so the concepts are more balanced.
- **Annotation pipeline:**
  1. An objectness detector, trained on every detection dataset they had, tries to find all the objects in the image.
  2. SAM gives a mask for each of them.
  3. A fine-tuned Qwen2.5-VL-7B then writes a label from coarse to fine, for example "animal, dog, a yellow dog". They fine-tuned it to always answer in this fixed format and to refuse boxes that are wrong, and the refused boxes are thrown away.
- **Size.** In the end they have 15M images and 330M boxes. They also add OpenImagesV6, Objects365 V2, V3Det, ImageNetBox and GoldG on top.
- **Ablation (Table 6, LVIS minival):**

  | Setting | AP |
  |---|---|
  | Full model | 47.3 |
  | Without coarse labels | 46.4 |
  | Without fine labels | 45.1 |
  | Without staged training | 45.5 |

  So yes, the label granularity matters, by roughly 1–2 AP. Dropping the fine labels hurts a bit more than dropping the coarse ones.

**Q3. What is WeDetect-Uni, and how is it different from a class-agnostic region proposal network?**
- **How it works.** They freeze the whole detector and only train one learnable objectness prompt. The paper calls it "kind of linear probing", which is a good way to think about it.
- **Difference from an RPN.** The important point is that the box embeddings of the top proposals are still class-specific. So later you can still classify them or match them with any text query, which a normal class-agnostic RPN box can't do.
- **Recall (Table 3, 300 proposals):**
  - WeDetect-Large-Uni gets AR50 95.3 on COCO, AR 65.1 on LVIS and AR 35.8 on PACO-LVIS.
  - For comparison, RPN-R50 only gets AR 31.4 on LVIS.

**Q4. What is the "object retrieval" task, how is it evaluated, and what does it show about image-level CLIP?**
- **Task.** You give a class name, and the model has to return every image in the database that contains that object, even very small ones (their example is cigarette butts).
- **Evaluation:**
  - They take a normal detection validation set as the database and use the class names as queries.
  - The ground truth for a query is just the set of images that have annotations of that class.
  - Precision, recall and F1 are computed for each class first and then averaged.
  - On LVIS they only report recall, because LVIS is a federated dataset and not every class is labelled in every image.
  - They reuse the COCO threshold on LVIS "to prevent setting a low threshold to get a high recall rate". We liked that they did this.
- **Results (Table 4, COCO F1):**

  | Method | F1 |
  |---|---|
  | OpenAI CLIP ViT-L/14-336 | 46.4 |
  | HQ-CLIP | 52.2 |
  | FG-CLIP2 | 57.7 |
  | WeDetect-Large-Uni | 83.6 |

- *(Our reading)* For us this was probably the most useful table in the paper. A whole-image embedding is quite bad at telling whether an object is present, especially a small one. Our selector makes exactly this decision with one whole-image RemoteCLIP embedding, so we should expect the same limit. It also fits with our M5 result, where local crops raised small-class selection recall from 0.22 to 0.31.

**Q5. How does WeDetect-Ref turn referring-expression comprehension into retrieval, and why is it faster than normal LLM detectors?**
- **How it works:**
  1. WeDetect-Uni gives the list of candidate boxes.
  2. For each box they take RoI features from Qwen3-VL's visual encoder and squeeze them into a single `<object>` token with a linear projector. The box position is added as an embedding.
  3. The image tokens, the user's query and all the object tokens go into the LLM together.
  4. A new binary classification head then gives each object token a yes/no score.
- **Training.** They use sigmoid focal loss with IoU as soft labels, and any proposal with IoU > 0.5 to a ground-truth box counts as positive. Training happens in three stages:
  1. only the projector, on 700K caption samples;
  2. the LLM and the projector, on around 1.7M region and image instruction samples;
  3. a classification stage on 4M detection and REC samples.
- **Why it is faster.** There is no next-token generation at all. All objects are scored in one forward pass instead of being written out one by one.
- **Results:**
  - WeDetect-Ref 4B averages 93.2 on RefCOCO/+/g at 5.3 FPS. Qwen3-VL 4B averages 86.6 at only 0.4 FPS, which is 13× slower.
  - According to the authors it is the first LMM to go above 50 AP on COCO (50.0).

**Q6. What do the ablations say about negative queries, and what does that mean for a frozen detector that is given absent class names?**
- **Ablation (Table 7):** when they remove the negative detection data, COCO AP drops from 47.5 to 42.1 and APs drops from 29.6 to 22.6. That is a big drop for small objects.
- **Why.** In their words, "recent LMMs tend to predict objects for every query, failing to reject negative queries". This is also why Qwen2.5-VL-7B only gets 17.7 AP on COCO.
- *(Our reading)* We think this is the same issue we had. If you ask a detector for a class that isn't in the image, it still gives confident boxes. WeDetect fixes it by training with negative queries. We can't retrain Grounding DINO, so we do the inference-time version: we remove the names that are probably not in the image.
  - In our failure analysis, 77% of the confident false positives with all 80 classes were classes that are never annotated in that image's dataset.
  - Our selection cut confident false positives by 76%.

**Q7. What are the limits of WeDetect when read as evidence for aerial detection?** *(our reading)*
- **Benchmarks.** Everything is on natural images: LVIS, COCO, COCO-O, ODinW13/35, RefCOCO and HumanRef. ODinW does have a few drone datasets mixed in, but they are not reported on their own, so we can't really tell how it does on aerial data.
- **Speed setting.** The FPS numbers are measured on COCO at 640 px (Tiny and Base) or 1280 px (Large). Our aerial images are bigger and much denser, so we would not expect the same speeds.
- **Training cost.** All their gains come from training a new model: 15M curated images, staged training and their own ConvNeXt CLIP. That is a very different budget from ours, since we don't train anything and keep the detector frozen.

---

## 2. LBP: Learning Background Prompts to Discover Implicit Knowledge for OVD (Li et al., CVPR 2024)


**Q1. What is wrong with how earlier distillation-based OVD methods treat the background?**
- **Single background class.** Methods like ViLD and BARON put every background proposal into one "background class" that has only one learnable embedding. The authors argue that this throws away a lot of information, because the background actually contains many unseen objects.
- **Uniform-dissimilarity assumption.** Earlier fixes, like BCE loss or DetPro's soft background loss, assume background proposals are "uniformly dissimilar to any foreground category". The problem is that this ignores how classes are related to each other.
- **Overfitting.** Most of the training signal comes from the fully labelled base classes, so the detector becomes biased towards them.
- **Need for names.** The methods that try to mine novel objects from the background usually need extra prior knowledge, "such as the names of novel categories". LBP wants to avoid that.

**Q2. How does the Background Category-specific Prompt (BCP) module find and represent background categories without knowing their names?**
1. They first train a class-agnostic RPN on the base classes and use it to get background proposals for all training images. These are filtered by RPN score (threshold θ) and NMS.
2. The CLIP image encoder embeds the remaining proposals, and they run k-means on these features to estimate how many hidden categories there are (n_o).
3. They add n_a extra categories on top, since overlapping features can make k-means guess too few.
4. Each of these n_o + n_a categories gets its own learnable context words in the prompt "a photo of {v_c} in the scene". There is also one more learnable sub-background embedding for anything that doesn't fit.
5. The loss tries to push each background proposal towards these background categories:
   - **L_bcp** maximises the sum of their probabilities;
   - **L_rlx** is used instead when that sum is below a small threshold γ, and it just spreads the probability evenly.

**Q3. How does Background Object Discovery (BOD) create pseudo-labels during training?**
1. They run k-means (with k = n_o) on the CLIP embeddings of the background proposals, and each cluster centre stands for one estimated category.
2. In every batch the background proposals are filtered by RPN score, and anything overlapping a ground-truth box is removed.
3. Each remaining proposal is labelled with the cluster that gives the highest softmax over cosine similarity.
4. Low-confidence labels are thrown out using a threshold. After that, per-class NMS and the RoI head clean up the boxes.
5. The loss L_bod handles positive and negative background proposals separately, and the negative part gets a small weight λ_bg.

The CLIP text and image encoders stay frozen the whole time.

**Q4. What is Inference Probability Rectification (IPR), and why is it needed?**
- **The problem.** At test time some of the learned background categories (C_O) can mean the same thing as a novel class c. Then c basically gets counted twice in the softmax denominator, once as a real class and once hidden inside the background terms, and its probability ends up too low. This part took us a while to understand, but the idea is simple once you see the equation.
- **The fix.** IPR shrinks each background category's term by (1 − Σ P(c″|c′)). P(c″|c′) is how much background category c′ overlaps with novel class c″, and they estimate it from the cosine similarity of the text embeddings, without looking at the image.
- **Ablation (Table 3, OV-COCO novel AP50):**

  | Setting | Novel AP50 |
  |---|---|
  | Baseline | 35.8 |
  | BCP alone | 34.3 (overlap hurts) |
  | BCP + IPR | 36.8 (+2.5 over BCP alone) |
  | All three modules | 37.8 |

  What we found interesting is that BCP alone actually makes novel classes worse. It only helps once IPR removes the overlap.

**Q5. Which datasets, splits and metrics are used, and what are the main results?**
- **OV-COCO:**
  - The split is 48 base and 17 novel classes, and the main metric is novel AP50.
  - LBP† gets 37.8 novel AP50 compared with 35.8 for BARON† (Table 1). The † means they trained with batch size 16, which helps against overfitting to the base classes.
- **OV-LVIS:**
  - The 866 frequent and common classes are base and the 337 rare classes are novel. The main metric is APr.
  - LBP‡ gets 24.1 box APr and 23.7 mask APr, against 23.2 / 22.6 for BARON‡ (Table 2). Here ‡ means they used DetPro-style learnable prompts.
- **Comparison settings.** The paper groups the methods into vanilla, caption-based, generalised and weakly supervised OVD. LBP belongs to the vanilla setting, but they compare against all four.

**Q6. Does the overlap argument carry over to Grounding DINO and to our vocabulary selection?** *(our reading)*
- **The general point carries over.** If two text entries describe the same object, they fight over it and the right one loses score. We saw this in LAE-80C: 59% of the classes our selector picked came from another source's label set, e.g. "Ferry" on an image whose label is "Passenger Ship".
- **The exact maths does not.** Grounding DINO doesn't have a single softmax over all classes. It uses sigmoid scores per token, so the competition happens in other places:
  - in the language-guided query selection (the top 900 queries are picked by similarity to the text);
  - in the cross-attention.
  So we can't just plug IPR's formula into Grounding DINO.
- **A future-work idea.** IPR does give us an idea that needs no training:
  1. measure how similar the candidate class names are using their text embeddings;
  2. merge or down-weight names that are near-duplicates before sending them to the detector.

**Q7. What are LBP's limitations from our point of view?** *(our reading)*
- **It needs training.** It retrains the detector, so we can't use it on a frozen model like ours.
- **Sensitive hyper-parameters.** There are quite a few settings to tune: the number of clusters from k-means plus n_a, and the two thresholds θ and γ.
- **Small gains.** Compared with BARON the improvement is around +0.1 to +2 points, which is not huge.
- **Evaluation scope.** They only test on natural images, OV-COCO and OV-LVIS.

---

## 3. GenerateU: Generative Region-Language Pretraining for Open-Ended Object Detection (Lin et al., CVPR 2024)


**Q1. What is "generative open-ended object detection", and how is it different from OVD and phrase grounding?**
- **OVD and phrase grounding** both need the categories or phrases to be written into the text prompt, so the user has to know beforehand what to look for.
- **Open-ended detection** gives the model no category information at inference at all. It has to find the objects and come up with their names by itself.
- **Motivation (from the paper):**
  - In practice users often don't know exactly which categories are in the test images.
  - A list written by hand can be ambiguous or incomplete. Their example list misses "chain link fence".
  - Building one huge label set that covers everything "would be complex and time-consuming". This is pretty much what our "all 80 classes" baseline is.

**Q2. What is GenerateU's architecture and training objective?**
- **Detector.** It's a class-agnostic Deformable DETR (6 encoder and 6 decoder layers, 300 queries, with Swin-T or Swin-L as backbone). It only learns foreground vs background, trained with binary loss, GIoU loss and L1 loss.
- **Language model.** Every query that Hungarian matching pairs with a ground-truth box is projected into the encoder of a FlanT5-base language model. The decoder then generates the object name, trained with the usual language-modelling loss.
- **Region-word alignment loss.** They add an extra BCE loss between the region features and frozen CLIP text embeddings, where the other words in the same batch act as negatives.
- **Training setup.** They use AdamW, a learning rate of 2e-4 for the detector and 3e-4 for the language model, and 16 A100 GPUs, which is way more compute than we have.

**Q3. Why does plugging a detector into a frozen multimodal LLM work poorly?**
- **Results (Table 2, zero-shot LVIS AP):**

  | Setting | AP |
  |---|---|
  | Frozen KOSMOS-2 (image encoder and language model frozen) | 14.3 |
  | KOSMOS-2 with a trainable image encoder | 21.9 |
  | Initialised from an LLM, language model frozen | 22.9 |
  | Full end-to-end training | 26.8 |

- **Reasons given by the authors:**
  - The image encoder was trained on 224×224 images, "which is insufficient for the object detection task".
  - A multimodal LLM that was trained only with whole-image captions "lacks understanding in both object localization and fine-grained object categories".
- *(Our reading)* Both of these problems apply to our selectors as well:
  - RemoteCLIP and CLIP ViT-B/32 also look at the whole image at 224 px;
  - our caption selector (VocAda-style) is built on whole-image captions.

**Q4. Which datasets are used, and how is label diversity increased?**
- **Detection data.** They use the object annotations of Visual Genome, 77,398 images. They deliberately skip VG's region captions, since those can describe several objects or even actions.
- **Image-text data.** Around 5M GRIT pairs from COYO-700M, with pseudo boxes that KOSMOS-2 generated using GLIP.
- **Pseudo-labels.** Captions name far fewer objects than an image actually contains, so they use an already trained GenerateU to label the missing objects.
- **Beam search.** They noticed that beam search gives synonyms and names at different levels for free. The default beam size is 3, and in Fig. 4 APr goes from 12.3 with beam 1 up to 20.0 with beam 3.

**Q5. How do the authors evaluate free-form names against a fixed label set?**
- **Name matching.** Every generated name is mapped to the closest dataset category using a fixed text encoder (CLIP text or BERT-large). The detector never uses this encoder; it is there "only for quantitative performance measurement".
- **METEOR.** They also score the text like in dense captioning, using AP over IoU thresholds 0.3–0.7 and METEOR thresholds 0–0.25.
- **Data.** LVIS minival (5k images), with fixed AP.
- **Results (Table 4):** the same predictions get 26.8 AP with CLIP matching and 21.7 AP with BERT matching. With METEOR it's 9.2.
- *(Our reading)* So just changing the evaluator moves the result by about 5 AP, with the same model. This made us more confident about reporting both standard and source-aware AP, and about fixing the protocol before running on the test set.

**Q6. What are the main quantitative results?**
- **Zero-shot LVIS (Table 1):**

  | Model | Training data | AP | APr |
  |---|---|---|---|
  | GenerateU Swin-T | VG only | 25.4 | 17.4 |
  | GenerateU Swin-T | VG + GRIT5M | 26.8 | 20.0 |
  | GenerateU Swin-L | VG + GRIT5M | 27.9 | 22.3 |
  | GLIP-T | O365, GoldG, CAP4M | 26.0 | 20.8 |

  Adding GRIT gives +3.6 APr and +1.4 AP, so the extra image-text data helps most on rare classes.
- **Zero-shot transfer (Table 3):** it gets 33.6 on COCO and 10.5 on Objects365, while a supervised model gets 46.5 / 25.6. So there is still a big gap there.
- **Region-word alignment loss (Table 5):** it adds +2.9 AP when training on VG only and +0.9 AP with VG + GRIT.

**Q7. Could open-ended generation replace vocabulary selection for aerial imagery?** *(our reading)*
- **We don't think it could do it directly, for three reasons:**
  - **Names still have to be mapped.** To score the outputs, or to use them in an application, the generated names have to be mapped back to some label set. In LAE-80C that label set has near-synonyms across sources, so the mapping itself would be messy.
  - **Training cost.** It needs full training on VG plus 5M pairs on 16 A100s. Our selector doesn't need any training.
  - **Domain.** Its language model learnt names of everyday objects, not aerial ones like "storage tank" or "bridge".
- **What we can take from it.** The idea of generating names and then mapping them, plus the synonym lists from beam search, could be another way to produce candidate classes for our selector.

---

## 4. Bridging the Gap between Object and Image-level Representations for OVD (Rasheed et al., NeurIPS 2022)


**Q1. What is the "gap between object and image-level representations" that the paper targets?**
- **Weak supervision is image-level.** Open-vocabulary detectors grow their vocabulary using CLIP, image-level labels or captions, and all of these describe the whole image, not single objects. As the authors put it, CLIP embeddings "do not perform well in locating object regions … since the CLIP model is trained with full scale images".
- **Labels are grounded by heuristics.** To turn an image-level label into a box, earlier work uses simple rules, like taking the max-score or max-size RPN box, or the whole image (Detic). The authors point out that these "can ignore potential objects".

**Q2. How does region-based knowledge distillation (RKD) work?**
1. They query a pre-trained multimodal ViT (MViT) with the text "all objects" to get class-agnostic proposals, and keep the top 5.
2. CLIP ViT-B/32 (224×224 input) embeds these crops offline, before training.
3. The detector's region embeddings are then trained to match the CLIP ones with two losses:
   - an L1 loss on each embedding;
   - an inter-embedding relationship loss, which compares the K×K similarity matrices of the CLIP and detector embeddings with a Frobenius distance.
4. The loss weights are β1 = 0.15 and β2 = 0.15 on COCO (0.25 on LVIS). They picked these with a random search on held-out data.

**Ablation (Table 7, COCO novel AP50):**

| Setting | Novel AP50 |
|---|---|
| Supervised base model | 1.7 |
| L1 loss on RPN proposals | 4.0 |
| L1 loss on MViT proposals | 16.7 |
| L1 + IRM losses | 21.2 |

The jump from RPN to MViT proposals was the most striking number for us. The loss is the same, only the proposals changed.

**Q3. How do they turn image-level labels into box supervision, and how does that compare with heuristics?**
- **Method.** For every image-level label, taken from COCO Captions (for COCO) or the 997-class ImageNet-21K subset (for LVIS), they:
  1. query MViT with "every {category}";
  2. keep the single top proposal as the pseudo box;
  3. train only the classification part on it (the L_pms BCE loss, with weight α = 0.1).
- **Comparison (Table 8, novel AP50):**

  | Method | Novel AP50 |
  |---|---|
  | Max-score RPN box | 15.9 |
  | Max-size RPN box | 25.9 |
  | Max-size MViT box | 28.9 |
  | Pseudo-box from MViT (theirs) | 30.4 |

  So using a class-specific query is better than any of the heuristic box choices.

**Q4. Why is a weight-transfer function needed instead of simply adding the two losses?**
- **Naive combination fails to add up.** When they just put RKD and PIS together, they get 31.5 novel AP50, barely above PIS alone (30.4). The authors say the two methods "compete with each other".
- **The weight-transfer function.** Instead, they generate the image-level projection W_P from the frozen distillation projection W_D. It's a 2-layer MLP (hidden size 512, LeakyReLU with slope 0.1) with an extra skip MLP (hidden size 1024).
- **Result.** With this, novel AP50 goes up to 36.6 and base AP50 stays at 54.0 (Table 2).
- *(Our reading)* We ran into something similar. When we added crop evidence or uncertainty retention on top of the global score, detection AP didn't go up, even though each idea looked useful by itself.

**Q5. What datasets, protocols and training setup are used?**
- **COCO:**
  - 48 base and 17 novel classes, in the generalised setting where the classifier has both base and novel classes. The metric is box AP50.
  - The detector is a Faster R-CNN R50, trained 1× on base, 1× with RKD and 1× with PIS plus weight transfer, so 3× in total. Learning rate 0.02, batch size 16, temperature 50.
- **LVIS:**
  - 866 base and 337 rare classes, with mask AP averaged over 3 runs.
  - They use Mask R-CNN, and also a CenterNetV2 version so they can compare with Detic.
- **Cross-dataset:** COCO, OpenImages and Objects365.
- **Leakage control.** They retrained MViT on a filtered version of LMDet so it never sees the novel or rare classes used in evaluation. We thought this was a careful thing to do.
- **Compute.** 8 A100 GPUs.

**Q6. What are the main results, and what does the "strict OVD" experiment show?**
- **COCO:** novel AP50 is 36.6, against 28.4 for Detic and 27.6 for ViLD. With the longer 8× schedule it gets 36.9 novel and 56.6 base.
- **LVIS:**
  - After 96 epochs APr is 21.1, while ViLD gets 16.1 after 384 epochs.
  - On Detic's CenterNetV2 baseline, APr goes to 25.2 compared with 24.6.
- **Cross-dataset AP50:** COCO 56.6, OpenImages 42.9, Objects365 22.3, all a bit better than Detic.
- **Strict OVD.** When the image-level labels come from 5K random ImageNet-21K classes (5× the vocabulary), AP falls to 21.8, which is slightly below ViLD's 22.5. If they use only the LVIS base classes, AP is 21.71, compared with 22.75 with the 997 classes.
- **Takeaway.** From this we understood that the results depend quite a lot on how big the vocabulary is and how well it matches the target classes.

**Q7. What does this paper imply for a CLIP-based vocabulary selector?** *(our reading)*
- **Bias.** If a selector scores the whole image with CLIP, it gets CLIP's scene-centric bias too. It will prefer large, scene-level classes and miss small objects in dense scenes. This matches our result that the correct class was kept for only 26% of ground-truth objects, because the dense FAIR1M and xView scenes hold most of the objects.
- **Region evidence needs good proposals.** Going from RPN to MViT proposals (4.0 → 16.7 novel AP50) makes us think our crop-based selector was held back by using fixed crops instead of real object proposals.

---

## 5. CondHead: Learning to Detect and Segment for OVD (Wang, CVPR 2023)


**Q1. What limitation of earlier OVD detectors does CondHead address?**
- **Class-agnostic heads.** Earlier methods (OVR-CNN, ViLD, RegionCLIP, DetPro) mostly worked on classifying the proposals. The box regression and mask segmentation were left class-agnostic.
- **Why that is a problem.** Class-agnostic heads work for any class, but the author says they "offer limited capacity to learn category-specific knowledge like object shape".
- **Why class-wise heads are not an option.** You would need box and mask annotations for the novel classes to train class-wise heads, and in OVD you don't have those.
- **The intuition (Fig. 1).** Box and mask knowledge can be shared between similar classes, for example a chicken can help with an ibis and a toothbrush can help with a hairbrush. We liked this figure, it made the idea easy to follow.

**Q2. How is CondHead built?**
- **Dynamically aggregated head:**
  - There are H expert heads. The box experts are 2 fully connected layers and the mask experts are 3 convolutional layers.
  - A small network A takes the class embedding s_c and outputs H weights. These are normalised with a softmax, and the experts' parameters are averaged with those weights into one head.
  - Box and mask heads each have their own weight network.
- **Dynamically generated head.** Another network D directly generates the parameters of a small head (1 fully connected layer for boxes, 1 convolutional layer for masks) from s_c. According to the paper it captures things like aspect ratio and colour.
- **Combining the two.** The two predictions are mixed with weights λ and μ, and values between 0.5 and 0.7 work best.
- **Training.** They anneal the softmax temperature: at the start all experts get trained about equally, and later they are allowed to specialise.
- **Size.** The hidden dimension is 256, and adding more than about 40 experts doesn't help much more.
- **Inference.** At test time the heads are conditioned on the embedding of the target class, s_m.

**Q3. Which datasets, baselines and protocol are used?**
- **Datasets.** COCO and LVIS, using the base/target splits from the OVR-CNN and RegionCLIP preprocessing.
- **Cross-dataset.** They train on LVIS and then test on PASCAL VOC, COCO and Objects365, just by swapping in the target dataset's vocabulary.
- **Baselines.** OVR-CNN, ViLD and RegionCLIP, picked because their semantic-visual alignment gets stronger in that order. Each baseline is re-run with a class-agnostic mask head (marked * in the tables) so the comparison with CondHead is fair.

**Q4. What are the main results and the cost?**
- **COCO (Table 2), novel AP50 gains:**

  | Baseline | Gain | With CondHead |
  |---|---|---|
  | OVR-CNN | +1.4 | |
  | ViLD | +2.1 | |
  | RegionCLIP | +2.4 | 33.7 |

  Mask AP goes up by similar amounts.
- **LVIS (Table 3):** around +2 to +3 APr for both boxes and masks. With RegionCLIP RN50x4, APr goes from 22.1 to 25.1.
- **Cross-dataset (Table 4), AP50 gains:** +2.4 on VOC, +3.5 on COCO and +2.2 on Objects365.
- **Cost (Table 7):** only 193.5 → 195.8 GFLOPs, which is about 1.1% more computation.
- **Shape priors.** When they also add ShapeMask priors, the mask APr gain goes up to +3.5 / +3.6 (Table 6).

**Q5. Why does the gain grow with stronger base models, and why is nearest-class transfer not enough?**
- **Alignment.** CondHead depends on the class text embedding matching the visual features well. When the alignment is better (RegionCLIP > ViLD > OVR-CNN), the conditioning carries over better to new classes.
- **Naive "ClassWise" baseline (Table 5).** They also tried the simple approach: train class-wise heads on the base classes, then for a novel class use the head of the most similar base class. It only gives +0.3 to +0.5 APr, because base and novel classes still differ in appearance and context.
- *(Our reading)* We saw the same thing with alignment in our project. The gain was large where RemoteCLIP knows the images (DIOR and DOTA, source-aware AP50 0.035 → 0.120) and there was basically none where it doesn't (FAIR1M and xView, 0.008 → 0.009).
- **Table inconsistency we noticed.** Table 5 gives CondHead on RegionCLIP as 17.9 (+2.9) box APr. But 17.0 + 2.9 = 19.9, and Table 7 also says 19.9, so we think 17.9 is a typo.

**Q6. What does the "language description" experiment show, and why does it matter for vocabulary choice?**
- **Results (Fig. 4):**
  - If you give the wrong class text, the box regression gets worse. In their example a sunflower proposal gets pulled towards a carrot.
  - If you give a hand-tuned description it gets better. Using "snowboard" and "motorcycle" as the description improves the box for a snowmobile.
- **The authors' conclusion.** They say better performance "could be achieved by simply tuning the semantic descriptions during inference".
- *(Our reading)* For us this was the most relevant part of the paper. In a detector that is conditioned on text, the class names decide where the boxes go, not just what they are called. Grounding DINO depends on the text even more (feature fusion and language-guided query selection), so removing irrelevant names should also improve the localisation. Our vocabulary selection is one kind of the inference-time text tuning the author suggests.

---

## Cross-paper question

**Which of these papers does inference-time, per-image vocabulary selection for a frozen detector?**
- **None of them.**
  - WeDetect, LBP, Rasheed et al. and CondHead all retrain a detector, and all of them assume someone already gives the class list.
  - GenerateU gets rid of the list, but it needs full training, and for evaluation it still has to map the names back to a list.
  - The only things done at inference time are LBP's IPR and CondHead's description tuning, and neither of them chooses which classes to ask for.
- **None of them evaluates on a remote-sensing benchmark.**
- This supports the gap stated in `reports/M0_gap_hypotheses_novelty.md`.
- See `annotated_papers_synthesis.md` for the combined picture.
