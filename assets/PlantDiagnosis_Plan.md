# SynapseCore: Evidence-Grounded Plant Disease Understanding

## Project summary

**SynapseCore** is a two-stage plant-diagnosis system. It discovers likely individual leaves in a field image, predicts each leaf's crop and disease, shows model-relevant visual evidence, and then uses a vision-language model (VLM) to explain the judgement from that evidence.

- **V1 — perception:** leaf discovery, per-leaf crop/disease prediction, and visual evidence.
- **V2 — reasoning:** evidence-grounded explanation and question answering.

The first viable project is deliberately modest and scientifically honest: it produces **crop name + disease name**, leaf-level results where possible, and **model-derived evidence regions**. It does not claim perfect disease segmentation.

---

## 1. Problem statement

Given a photograph containing one or more leaves, possibly with clutter, shadows, overlap, and complex backgrounds, build a system that can:

1. find and separate likely leaf instances;
2. predict a crop and disease for each usable leaf;
3. identify the image areas that influenced the disease prediction; and
4. explain the final judgement using the original image, leaf crops, evidence regions, and structured predictions.

A desired image-level summary is:

> **Tomato — late blight.** Two likely affected leaves were found out of three leaf candidates. The prediction is supported by high-importance discoloured regions on those leaves.

“High-importance” means evidence used by the model. It does **not** mean an exact disease-pixel mask.

---

## 2. V1 and V2 goals

### V1 — SynapseCore Vision

\`\`\`text
Input image
  -> leaf discovery and separation
  -> individual leaf crops
  -> shared crop + disease model
  -> Grad-CAM evidence maps
  -> structured per-leaf and image-level result
\`\`\`

V1 answers:

- How many plausible leaves were found?
- What crop and disease are predicted for each usable leaf?
- How confident is the prediction?
- Which areas influenced the prediction?

### V2 — SynapseCore Vision-Language

\`\`\`text
Original image + leaf crops + evidence crops + V1 JSON + user question
                                  -> VLM
                                  -> grounded explanation
\`\`\`

V2 explains V1's evidence. The VLM is not the sole diagnostic authority.

### Initial non-goals

- Do not claim a supervised leaf detector trained from PlantVillage/PlantDoc alone.
- Do not claim disease segmentation from PlantVillage/PlantDoc alone.
- Do not call Grad-CAM a disease mask.
- Do not allow a free-form VLM to replace the vision model's prediction.

---

## 3. Data truth and dataset roles

| Dataset | Available in this plan | Exact role | Must not be assumed |
|---|---|---|---|
| **PlantVillage** | Image, crop name, disease label; controlled background and usually one leaf | Clean visual foundation; classification baseline and pretraining | Leaf boxes, leaf masks, disease masks |
| **PlantDoc** | Image, crop name, disease label; field/complex images and possibly multiple leaves | Field robustness, domain adaptation, real-world evaluation | Leaf boxes, leaf masks, disease masks |
| **AgroBench** | Agricultural multimodal benchmark resource, used only under its released task definitions | V2 benchmark/adaptation for language-grounded agricultural understanding | Any V1 detection or segmentation annotation not explicitly confirmed |
| **Optional manually annotated PlantDoc subset** | A deliberately created subset with leaf boxes or masks | Train and evaluate a leaf detector/segmenter | Disease masks unless separately annotated |

The core available supervision is:

\`\`\`text
image -> crop label + disease label
\`\`\`

Neither PlantVillage nor PlantDoc supplies, in the stated setup:

\`\`\`text
leaf bounding box
leaf instance mask
disease-region mask
\`\`\`

Therefore, the two datasets alone cannot honestly supervise leaf detection or disease segmentation. They can supervise crop/disease classification.

### Canonical label taxonomy

Before training, create a versioned mapping:

\`\`\`text
source dataset label -> canonical crop -> canonical disease -> include/exclude reason
\`\`\`

Only reliably matched crop–disease classes should be used in PlantVillage-to-PlantDoc transfer experiments. Never silently merge uncertain labels.

---

## 4. End-to-end architecture

\`\`\`mermaid
flowchart TD
  I[Plant image] --> L[Leaf discovery]
  L --> F[Leaf filtering and quality checks]
  F --> C[Leaf crops]
  C --> E[Shared visual encoder]
  E --> CH[Crop head]
  E --> DH[Disease head]
  DH --> G[Grad-CAM evidence]
  CH --> S[Structured V1 result]
  DH --> S
  G --> S
  I --> P[Original image context]
  C --> V[Leaf and evidence crops]
  P --> X[Evidence package]
  V --> X
  S --> X
  Q[User question] --> M[VLM]
  X --> M
  M --> O[Grounded explanation]
\`\`\`

Markdown-safe view:

\`\`\`text
Image -> leaf proposals -> leaf filtering -> individual leaf crops
                                              |
                                              v
                                       shared visual encoder
                                          /             \
                                     crop head      disease head
                                          \             /
                                           predictions -> Grad-CAM
                                                              |
Image + crops + evidence + structured prediction + question -+-> VLM -> explanation
\`\`\`

---

## 5. V1 component design

### 5.1 Leaf discovery

Field images can contain multiple leaves, soil, stems, shadows, and non-leaf objects. Whole-image classification may mix evidence from different leaves, so V1 should attempt leaf discovery first.

**First baseline: pretrained segmentation proposals**

\`\`\`text
PlantDoc image -> SAM / SAM 2 candidate regions -> leaf/non-leaf filtering -> leaf crops
\`\`\`

SAM/SAM 2 proposes plausible regions; it does not inherently know which regions are leaves. Filtering may use geometry/size checks, quality checks, and a lightweight leaf-vs-non-leaf classifier. These assumptions must be tested on reviewed images.

**Later, stronger route: annotate a PlantDoc subset**

Annotate a modest subset with leaf bounding boxes (or instance masks if practical), then train a detector such as YOLO:

\`\`\`text
Annotated PlantDoc subset -> YOLO leaf detector -> leaf boxes -> crop/classification crops
\`\`\`

This deliberately purchases the missing supervision. It is not something PlantVillage or unannotated PlantDoc already supplies.

### 5.2 Shared crop + disease visual model

Use one encoder and two heads rather than two unrelated networks. Crop identity and disease appearance are related.

\`\`\`text
leaf image x
     |
     v
F = E(x)  shared visual encoder
     |------------------|
     v                  v
crop head           disease head
\`\`\`

Mathematically:

\[
F=E(x), \qquad \hat y_{crop}=H_{crop}(F), \qquad \hat y_{disease}=H_{disease}(F)
\]

With cross-entropy losses:

\[
L=\lambda_cL_{crop}+\lambda_dL_{disease}
\]

Tune \(\lambda_c\) and \(\lambda_d\) on validation data. A later crop–disease compatibility constraint may mask impossible combinations, but raw and constrained results should both be logged.

### 5.3 Grad-CAM evidence

For target disease score \(y^c\) and convolutional feature maps \(A^k\), Grad-CAM computes:

\[
\alpha_k^c = \frac{1}{Z}\sum_i\sum_j\frac{\partial y^c}{\partial A^k_{ij}}
\]

\[
L^c_{GradCAM}=ReLU\left(\sum_k\alpha_k^cA^k\right)
\]

Pipeline:

\`\`\`text
disease classifier -> Grad-CAM heatmap -> normalize -> threshold -> connected components -> evidence regions
\`\`\`

Correct wording: “These regions contributed strongly to the model's disease prediction.” Incorrect wording: “These are exactly diseased pixels.”

### 5.4 V1 structured output

\`\`\`json
{
  "num_leaf_candidates": 3,
  "leaves": [
    {
      "leaf_id": 1,
      "crop": "tomato",
      "crop_confidence": 0.96,
      "disease": "late_blight",
      "disease_confidence": 0.91,
      "evidence_regions": [
        {"bbox_xyxy": [120, 80, 250, 190], "importance": 0.93}
      ],
      "warnings": []
    }
  ],
  "image_summary": {
    "dominant_crop": "tomato",
    "disease_summary": "late_blight",
    "affected_leaf_ids": [1]
  }
}
\`\`\`

Use “leaf candidate” when discovery is uncertain.

---

## 6. Training and data flow

\`\`\`text
PlantVillage (clean labels) -> initialize shared crop/disease model
                                      |
PlantDoc (field labels) ----> fine-tune / domain-robust training
                                      |
                                      v
                           held-out PlantDoc evaluation

PlantDoc -> SAM/SAM 2 proposal baseline
Optional annotated PlantDoc subset -> train/evaluate YOLO leaf detector

V1 image + crops + evidence + JSON -> V2 evidence package
AgroBench -> V2 benchmark/adaptation only where released labels permit
\`\`\`

### PlantVillage

Use PlantVillage to learn relatively clean crop and disease patterns, choose initial architectures, and create a controlled baseline.

### PlantDoc

Use PlantDoc as the reality check: complex background, illumination changes, occlusion, varied poses, and multiple leaves. Compare PlantVillage-only transfer with PlantDoc fine-tuning and mixed-domain training using a locked field test set.

### AgroBench

AgroBench belongs to V2. Use it only for its documented multimodal/language tasks, released splits, and permitted adaptation/evaluation. It does not turn PlantDoc into a detector or segmentation dataset. Record the exact task, labels, split, and licence in a dataset card before use.

### Optional annotated PlantDoc subset

Write annotation rules for leaf instances: partial/occluded leaf policy, overlap policy, and what counts as a leaf. Reserve a separate annotated test subset. Disease masks require separate annotation; leaf boxes do not justify a disease-segmentation claim.

A crucial caution: an image-level PlantDoc label does not guarantee that every generated leaf crop has that same disease. Per-leaf pseudo-labeling must be audited rather than treated as certain ground truth.

---

## 7. Model candidates

| Component | Candidate models | Why |
|---|---|---|
| Leaf proposals | SAM 2 / SAM | Fastest way to form a no-annotation baseline |
| Learned leaf detection | YOLO family | Practical box detector and simple evaluation |
| Shared encoder | ResNet-50, EfficientNet, ConvNeXt-Tiny, small ViT | Strong baselines across compute budgets |
| Prediction heads | Two linear/MLP heads | Clear multi-task formulation |
| Evidence | Grad-CAM / Grad-CAM++ | Class-specific visual evidence |
| VLM | Small open multimodal instruction model fitting available hardware | Evidence-grounded explanation, not direct diagnostic replacement |

Start with a CNN such as ResNet-50 or EfficientNet: it is easy to train and inspect with Grad-CAM. Add transformers after a reliable baseline.

---

## 8. V2 VLM integration

The VLM receives:

1. original image for context;
2. numbered leaf crops;
3. Grad-CAM overlays or evidence crops;
4. V1 JSON: predictions, confidence, leaf count, warnings, coordinates;
5. the user question.

\`\`\`text
original image + leaf crops + evidence overlays + V1 facts + question
                           -> constrained VLM prompt
                           -> explanation citing leaf IDs/evidence
\`\`\`

Prompt guardrails:

- use the supplied V1 crop/disease results rather than inventing a diagnosis;
- identify relevant leaf IDs and evidence regions;
- describe evidence cautiously;
- disclose low confidence and discovery warnings;
- never describe Grad-CAM as exact segmentation;
- avoid unsupported treatment and safety advice.

Example:

> Leaf 1 was predicted as tomato with late blight confidence 0.91. Its strongest model evidence is in the highlighted irregular dark/discoloured area. This is model evidence, not an exact disease map. Other leaf candidates are assessed separately.

Evaluate V2 against factual agreement with V1 JSON, correct leaf/evidence references, uncertainty disclosure, hallucination rate, and human-rated clarity/usefulness. Compare it with an original-image-only VLM.

---

## 9. Experiments and their purpose

| # | Experiment | Procedure | Why it matters |
|---|---|---|---|
| 1 | Clean classification baseline | Train shared model on PlantVillage; evaluate on PlantVillage | Validate taxonomy, preprocessing, and basic model |
| 2 | Controlled-to-field transfer | Train on PlantVillage; test on matched PlantDoc classes | Measure the real domain gap |
| 3 | Field adaptation | Compare PlantDoc-only, PlantVillage-pretrained then PlantDoc-fine-tuned, and mixed-domain training | Find how clean data helps field performance |
| 4 | Shared vs separate models | Multi-task encoder versus separate crop/disease networks | Test whether joint learning improves quality or efficiency |
| 5 | Leaf discovery baseline | SAM/SAM 2 proposals plus filtering; manually review a set | Test no-annotation leaf discovery honestly |
| 6 | Learned leaf discovery | Annotate PlantDoc subset; train YOLO; compare with SAM/SAM 2 | Determine whether annotation yields a better measurable solution |
| 7 | Whole image vs per-leaf | Compare whole-image classification with classification of discovered leaf crops | Test the central separation-first design |
| 8 | Evidence sanity | Grad-CAM/Grad-CAM++ overlays, occlusion/deletion checks, reviewed examples | Detect background shortcuts and test faithfulness |
| 9 | VLM grounding | Image-only VLM versus V1 evidence package | Test whether structured evidence improves factual explanations |

For a constrained V1, prioritize 1–5 and 7. Experiment 6 is the highest-value detection extension; 9 is V2.

---

## 10. Evaluation

### Crop and disease prediction

- accuracy where class balance permits;
- macro precision, recall, F1;
- per-class confusion matrices;
- top-k accuracy if a ranked UI is used;
- calibration (for example ECE and reliability plots);
- joint crop-and-disease correctness;
- PlantVillage-to-PlantDoc performance drop on matched classes.

### Leaf discovery

Use these only on manually annotated or otherwise verified leaf data:

- AP/mAP at stated IoU thresholds for box detection;
- instance precision, recall, F1;
- leaf-count error;
- mask IoU/Dice only if actual masks were annotated;
- inference latency.

Never report detector mAP or mask IoU without ground truth.

### Evidence

- predeclared qualitative review set;
- background-attention rate;
- occlusion/deletion faithfulness tests;
- optional reviewer rubric: plausible leaf tissue, plausible symptom region, background shortcut, unclear.

Do not compute disease-mask IoU for Grad-CAM unless disease masks are later collected.

### VLM

- factual consistency with V1 JSON;
- correct leaf ID/evidence references;
- unsupported-claim rate;
- appropriate uncertainty disclosure;
- human ratings of clarity and usefulness.

---

## 11. Split and leakage rules

1. Split before augmentation; keep all variants of one source image in its original split.
2. Lock a PlantDoc field test set before model selection.
3. Avoid near-duplicate photographs/leaves across splits where possible.
4. Version the canonical taxonomy, exclusions, seeds, preprocessing, and checkpoints.
5. Keep V2 evaluation images/questions out of VLM adaptation data.
6. Report all choices that alter the matched cross-dataset class set.

---

## 12. User-visible outputs

### V1

- original image with leaf boxes/masks;
- numbered leaf crops;
- crop and disease labels with confidence;
- Grad-CAM overlay/evidence boxes;
- image-level summary;
- warnings for low confidence, overlap, poor crop quality, or background evidence.

### V2

- answer to questions such as “Why was this predicted as late blight?”;
- explanation grounded in leaf IDs and evidence;
- clear uncertainty statement;
- inspectable V1 structured record.

---

## 13. Constraints and required honesty

| Constraint | Response |
|---|---|
| No leaf annotations | Pretrained proposals first; manually annotate a subset for trained detection |
| No disease masks | Grad-CAM evidence, not segmentation |
| Difficult field domain | Measure transfer and fine-tune/evaluate on PlantDoc |
| Multiple leaves | Diagnose per discovered leaf where confidence permits |
| VLM hallucination | Constrained evidence package and factual evaluation |
| Class mismatch | Canonical taxonomy and matched-class protocol |
| Limited compute | CNN baseline, proposal model, small/quantized VLM inference |

---

## 14. Development ladder

### Step 0 — Prepare the data

Inventory images/labels, make the canonical taxonomy, define matched classes, make reproducible splits, and document exclusions.

### Step 1 — Build the clean baseline

Train the shared PlantVillage crop+disease classifier. Inspect errors and Grad-CAM examples.

### Step 2 — Measure the real problem

Test the same model on matched PlantDoc classes. The performance drop is the domain-shift baseline.

### Step 3 — Adapt to field conditions

Compare PlantDoc-only, PlantVillage-pretrained then PlantDoc-fine-tuned, and mixed-domain training.

### Step 4 — Add leaf discovery

Prototype SAM/SAM 2 proposals, filtering, and leaf-crop extraction. Show candidate uncertainty explicitly.

### Step 5 — Make discovery measurable

If proposals are inadequate, annotate a small PlantDoc subset and compare YOLO with SAM/SAM 2 using detection metrics, separation quality, latency, and annotation cost.

### Step 6 — Add evidence

Generate Grad-CAM evidence per disease prediction and run sanity checks.

### Step 7 — Deliver V1

Return leaf-level structured predictions and an image-level crop+disease summary. This is the first viable SynapseCore system.

### Step 8 — Add V2

Build evidence-grounded VLM prompting, then use AgroBench only within its documented scope and evaluate grounding against an image-only VLM baseline.

### Step 9 — Extend responsibly

Possible later work: leaf instance segmentation, separately annotated disease segmentation, stronger domain generalization, uncertainty calibration, and carefully curated grounded VLM adaptation.

---

## Final architecture in one view

\`\`\`text
                    SYNAPSECORE V1: VISION

Field image
   |
   v
Leaf proposals (SAM/SAM 2 baseline, or YOLO after annotation)
   |
   v
Leaf filtering and quality checks
   |
   v
Per-leaf crops
   |
   v
Shared encoder -----> crop head ------> crop + confidence
      |
      +-------------> disease head ---> disease + confidence
                                         |
                                         v
                                   Grad-CAM evidence
                                         |
                                         v
                                Structured V1 record

             SYNAPSECORE V2: VISION-LANGUAGE

Original image + leaf crops + evidence overlays + V1 record + question
                                  |
                                  v
                   constrained evidence-grounded VLM
                                  |
                                  v
         explanation tied to leaves, evidence, and uncertainty
\`\`\`

## Success criteria

V1 succeeds when, on held-out field images, it returns reproducible and inspectable leaf candidates, crop/disease predictions, calibrated confidence, and Grad-CAM evidence honestly labelled as model evidence.

V2 succeeds when its explanations are more factually grounded, uncertainty-aware, and useful than an image-only free-form VLM response.

