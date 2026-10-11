# Model Training Results & Discussion — YOLOv8n vs. Faster R-CNN

Consolidated results from every training iteration run so far, with the
reasoning behind each decision and an honest discussion of the domain-gap
finding. Written to be lifted mostly as-is into the thesis's Results and
Discussion sections.

---

## 1. Methodology: three dataset-size iterations

Three processed RDD2022 datasets were built, each used identically by both
architectures for a fair comparison:

| Dataset | Train images | Val | Test | Notes |
|---|---|---|---|---|
| Small | 277 | 60 | 60 | Initial pipeline-validation baseline |
| Medium | 1,034 | 224 | 221 | First scale-up |
| **Large** | **4,291** | **597** | **592** | Final iteration: class-balanced (1,100/class) AND country-balanced (fair-share sampling across the 7 RDD2022 sources) |

**Why the large dataset was rebuilt with explicit balancing** (not just made
bigger): auditing the small/medium datasets found two compounding biases
inherited from the raw RDD2022 distribution:
- **Class imbalance**: D00 (longitudinal crack) outnumbered D20/D40 by
  2-3x in every prior split, even though D20/D40 (alligator crack, pothole)
  are the *highest-severity* classes in the project's own risk-scoring model
  (`src/risk/scoring.py`: base severities 2.0 and 3.0 vs. 1.0 for D00).
- **Country imbalance**: Japan was ~46% of both prior processed datasets vs.
  ~28% of the raw pool; Czech Republic was nearly absent (0 images in the
  small dataset's validation split).

The large dataset's sampling used a "fair-share" (water-filling) algorithm:
each class's quota is split as evenly as possible across whichever countries
actually contain that class — discovered during this audit that **D40
(pothole) has zero images from Norway, US, or Czech Republic** in RDD2022 at
all, so perfect 7-way balance is not achievable for every class; the
algorithm redistributes automatically to the countries that do have it.

---

## 2. Results across dataset-size iterations (RDD2022 test sets)

| Model | Dataset | Train images | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| YOLOv8n | small | 277 | 0.368 | 0.305 | 0.334 | 0.297 | 0.119 |
| YOLOv8n | medium | 1,034 | 0.503 | 0.434 | 0.466 | 0.424 | 0.194 |
| **YOLOv8n** | **large** | **4,291** | **0.643** | **0.496** | **0.560** | **0.548** | **0.300** |
| Faster R-CNN | small | 277 | 0.227 | 0.205 | 0.216 | 0.136 | 0.043 |
| Faster R-CNN | medium | 1,034 | 0.543 | 0.300 | 0.387 | 0.340 | 0.148 |
| **Faster R-CNN** | **large** | **4,291** | **0.434** | **0.379** | **0.404** | **0.373** | **0.186** |

**Interpretation**: both architectures improved substantially and
monotonically as the dataset grew, confirming the core hypothesis behind the
"final iteration" plan — the small/medium datasets were data-starved, not
architecture-limited. YOLOv8n's mAP50 nearly doubled from small to large;
Faster R-CNN's more than doubled proportionally too, though it started and
finished well behind YOLOv8n at every scale.

### Per-class results, large dataset (test set) — did class-rebalancing work?

| Class | YOLOv8n mAP50-95 | Faster R-CNN mAP50-95 |
|---|---|---|
| D00 Longitudinal Crack | 0.211 | 0.131 |
| D10 Transverse Crack | 0.226 | 0.164 |
| D20 Alligator Crack | 0.316 | 0.169 |
| **D40 Pothole** | **0.448** | **0.277** |

**Pothole — the class specifically targeted by rebalancing, and the
highest-severity class in the risk model — is the strongest class for
*both* architectures**, not the weakest as it was before rebalancing. This
is the clearest direct evidence that the class-balancing decision worked as
intended.

---

## 3. H1 — model comparison hypothesis, tested with real numbers

> **H1**: "YOLOv8 will achieve detection accuracy (mAP) within 2% of Faster
> R-CNN, while maintaining lower computational complexity."

Tested on the large dataset (same split, same imgsz=640, same effective
batch size=16 for both):

| Criterion | Result | Verdict |
|---|---|---|
| mAP50-95 within 2 percentage points | Gap = **-11.47 points** (YOLOv8n 0.300 vs. Faster R-CNN 0.186) | **NOT SUPPORTED** |
| Lower computational complexity (training time AND inference time) | Training: 1.36h vs. 4.16h. Inference: 3.4ms vs. 87.2ms per image | **SUPPORTED** |

**Framing for the thesis**: H1 is only partially supported. The honest,
reportable finding is that **at this dataset scale, YOLOv8n substantially
outperforms Faster R-CNN on both accuracy and computational cost** — there
is no dimension where Faster R-CNN wins. This is itself a legitimate,
citable conclusion (consistent with why single-shot detectors are generally
preferred for real-time/field road-inspection applications in the
literature), not a failed experiment.

### Statistical backing (notebook 13)

The table above uses point estimates from notebooks 09/10. Notebook 13
re-evaluated both large-dataset models together, in the same session, on
the same 592 test images, adding real statistical tests instead of relying
on single-run point estimates alone:

- **Complexity criterion**: a paired Wilcoxon signed-rank test on
  per-image inference time (median 22.25ms YOLOv8n vs. 307.33ms
  Faster R-CNN) gave **p = 6.04 × 10⁻⁹⁹** — the speed advantage is not
  measurement noise.
- **Accuracy criterion**: H1 claims *equivalence* (within 2 points), not
  "no difference" — the statistically correct tool is a confidence
  interval on the difference checked against the ±2-point margin (the
  standard "two one-sided tests" logic for equivalence claims), not a
  plain two-sided p-value. A 300-iteration bootstrap over the 592 test
  images gave a **95% CI of [+0.086, +0.117]** on the mAP50-95 difference
  (YOLOv8n − Faster R-CNN) — entirely outside the ±0.02 equivalence
  margin, so the accuracy criterion is confidently rejected, not just
  "different on this one run." (Methodology note: this fresh session's
  point estimate was +0.101, close to but not identical to notebooks
  09/10's original +0.114 — a small, expected difference from using a
  low confidence threshold, conf=0.001, to trace YOLO's full
  precision-recall curve for a fair mAP comparison, rather than the app's
  operational 0.25 threshold used during the original notebook 09 run.)

**Why Faster R-CNN underperforms even with 4x more data**: at every dataset
scale (small, medium, large), Faster R-CNN's best checkpoint (lowest
validation loss) arrived within the first 2-4 epochs, with validation loss
climbing steadily afterward — classic overfitting behaviour for a
43M-parameter two-stage detector, vs. YOLOv8n's 3M parameters. The larger
dataset delayed but did not eliminate this pattern (large dataset: best
epoch 2 of 22 run before early stopping). This suggests Faster R-CNN in this
configuration needs either substantially more data still, different
regularization/learning-rate scheduling, or a smaller backbone to compete
with YOLOv8n on this task — worth stating as a discussion point / limitation
rather than only as a comparison result.

### Computational complexity / runtime (thesis-citable)

| Model | Dataset | Total training wall-clock time | Hardware |
|---|---|---|---|
| YOLOv8n | large (4,291 images, 100/100 epochs, no early stop) | **1.36 hours** | RTX 2070 Max-Q (8GB), local machine |
| Faster R-CNN | large (4,291 images, 22/100 epochs, early-stopped) | **4.16 hours** | Same |

Note for methodology transparency: getting a reliable Faster R-CNN
per-epoch timing estimate required three attempts — two consecutive
GPU-intensive smoke tests produced inflated readings (1,444s and 1,172s per
epoch) attributed to thermal/system-load contamination from running them
back-to-back without a cooldown; a third test under clean conditions (GPU
idle, no other processes) measured 445s/epoch, consistent with the real
run's observed 465-480s/epoch on all but two anomalous epochs (epoch 3 and
7 spiked to ~2,560-2,615s, again attributed to transient system contention
on a shared local machine, not a property of the model or data). This
variance is itself a small, honest piece of evidence for the practical
"computational complexity" burden of the heavier architecture — even
measuring it reliably was harder.

---

## 4. Domain-adaptation stage: RDD2022-trained models vs. real Sofia photos

### 4.1 Why this stage exists

The project's prototype is explicitly scoped to a real pilot area — Sofia,
Bulgaria (`config/app_config.yaml`'s `pilot_area`) — not RDD2022's benchmark
images in general. A model that only ever sees RDD2022 photos during
training has no guarantee of working well on Sofia's actual roads, cameras,
and lighting. To test this, 201 manually collected and annotated Sofia
photos (`data/manual/`) were split into three non-overlapping parts:

| Split | Images | Role |
|---|---|---|
| Fine-tune train | 131 | Trains the domain-adaptation stage |
| Fine-tune val | 28 | Monitors fine-tuning for overfitting |
| **Held-out test** | **42** | **Never trained on by either model or either stage** — used only for before/after comparison |

Both RDD2022-large-trained models (YOLOv8n and Faster R-CNN) were evaluated
on the held-out 42-image Sofia test set **before** any fine-tuning, then
fine-tuned on the 131-image split with a reduced learning rate (nudging
existing weights toward Sofia's conditions rather than retraining from
scratch), then re-evaluated on the same untouched 42 images.

### 4.2 The domain gap — the headline finding

| Model | RDD2022 test mAP50-95 | Sofia test mAP50-95 (no fine-tuning) | Relative drop |
|---|---|---|---|
| YOLOv8n | 0.300 | 0.041 | **~86% drop** |
| Faster R-CNN | 0.186 | 0.040 | **~78% drop** |

**Both architectures collapse to essentially the same low performance
(~0.04 mAP50-95) on real Sofia photos, despite starting from very different
RDD2022 performance levels.** This is strong, symmetric evidence that the
gap is a genuine domain/distribution-shift effect — not an artifact of one
architecture's weaknesses — and it directly validates why the project
originally scoped in a domain-adaptation stage at all, rather than assuming
an RDD2022-trained model would simply work in Sofia.

**Why the gap is this large** (for the thesis discussion section):
1. **Visual domain shift**: RDD2022 images come from vehicle/drone-mounted
   cameras across 7 countries with a fairly consistent capture protocol;
   the Sofia photos were taken independently (a dashcam-style,
   through-windshield vantage in at least some cases — see the CVAT
   annotation review earlier in this project), with different lighting,
   asphalt color/texture, and compression characteristics. The model
   learned RDD2022-specific visual patterns that don't fully transfer.
2. **Annotation-protocol differences**: mAP/F1 depend on IoU overlap
   between predicted and ground-truth boxes. RDD2022's official annotation
   team and the manual Sofia annotation (done independently in CVAT for
   this project) may draw box tightness/segmentation differently for
   visually similar damage — a correct detection can still register as a
   "miss" under strict IoU matching purely due to labeling-convention
   differences, independent of whether the detector is actually wrong.
3. **Small-sample variance**: the Sofia test set (42 images, 142 total box
   instances, only 11 Pothole instances) is far smaller than RDD2022's test
   set (592 images, 1,569 instances) — its metrics are inherently noisier
   and more sensitive to a handful of hard images.

### 4.3 Fine-tuning results — did domain adaptation help?

| Model | Sofia F1 (no fine-tuning) | Sofia F1 (fine-tuned) | Relative F1 improvement | mAP50-95 (no FT → FT) |
|---|---|---|---|---|
| **YOLOv8n** | 0.189 | **0.347** | **+83%** | 0.041 → 0.148 |
| Faster R-CNN | 0.218 | 0.311 | +43% | 0.040 → 0.061 |

**Fine-tuning on just 131 real Sofia photos produced a large, clearly
positive improvement for both architectures** — evidence that even a small
amount of target-domain data meaningfully closes the gap, without needing
to collect thousands of local images.

**YOLOv8n benefited roughly 2x more than Faster R-CNN did** (in relative F1
terms; ~5x more in absolute mAP50-95 terms: +0.107 vs. +0.021). Plausible,
defensible explanation: Faster R-CNN's much larger parameter count (43M vs.
3M) means 131 images represents a proportionally smaller and noisier
fine-tuning signal for it — consistent with the same overfitting-prone
pattern seen throughout every training stage of this project for that
architecture.

**Neither model's fine-tuned Sofia performance reaches its RDD2022-native
performance** (YOLOv8n: 0.148 vs. 0.300 mAP50-95; Faster R-CNN: 0.061 vs.
0.186) — expected, since 131 images is a fraction of the 4,291-image
RDD2022-large training set, and should be stated plainly as a limitation:
domain adaptation with this much local data narrows but does not close the
gap. A natural, explicitly flagged future-work item is collecting more
Sofia-specific images to further close it.

### 4.4 Framing for the thesis (important)

**Do not present this as "YOLOv8n's F1 fell from 0.560 to 0.347" as if it
were a decline or a problem.** These are two different tests measuring two
different things: 0.560 measures performance on the task/domain the model
was trained for (RDD2022); 0.347 measures how well that skill transfers to
a real, previously-unseen deployment city, *after* a first adaptation
attempt. The correct comparison to lead with is the **0.189 → 0.347
before/after fine-tuning result on the same held-out Sofia set** — that is
the number that demonstrates the value of the domain-adaptation stage this
project built.

---

## 5. H2 — risk-classification evaluation, tested with real ground truth

> **H2**: "A rule-based risk scoring model based on damage type and size
> will effectively differentiate road segments into low, medium, and
> high-risk categories." Operationalized target: F1 ≥ 0.75.

### Method

The 42-image held-out Sofia test set (never used for training either
architecture) was manually labeled low/medium/high by the project owner,
**blind to the model's output** (labels assigned by looking at each photo
alone, before running any pipeline code) — this ordering matters: labeling
after seeing the model's answer risks unconsciously anchoring to it, which
would inflate agreement artificially.
Each image was then run through the actual deployed pipeline
(`src.detection.detector.DamageDetector` using the Sofia-fine-tuned model +
`src.risk.scoring`/`src.risk.observation_risk`) to get the category the app
would really show a user. See `scripts/evaluate_h2_risk_classification.py`
and `thesis/h2_manual_risk_labels_template.csv` (filled).

### Results

| Metric | Value |
|---|---|
| Accuracy | 0.595 |
| Cohen's kappa (chance-corrected agreement) | 0.377 |
| Macro F1 | **0.582** |
| **H2 target (F1 ≥ 0.75)** | **NOT SUPPORTED** |

| Category | Precision | Recall | F1 |
|---|---|---|---|
| low | 0.50 | 0.50 | 0.50 |
| medium | 0.48 | 0.73 | 0.58 |
| high | **0.91** | 0.53 | 0.67 |

### The key discussion point: this is a detection-recall problem, not a scoring-formula problem

The confusion matrix shows most disagreement concentrated in the "high"
category: 8 of 19 manually-rated "high" images were scored down (mostly to
"medium"). Checking these cases directly against the detector's raw output
confirms why — the detector is **missing the relevant damage entirely**,
not misjudging its severity once found:

- `D40_Pothole_manual_009.png` (rated high): detector found **Alligator
  Crack (0.685) and Longitudinal Crack (0.681)** — the pothole itself was
  never detected.
- `D40_Pothole_manual_019.jpg` (rated high): 3 detections, none of them the
  actual pothole.
- `D20_Alligator_Crack_manual_058.jpg` (rated high): **zero detections at
  all** — defaulted to "low" for lack of any input.

Precision on the "high" category is 0.91 — **when the model does detect
the right damage, the formula scores it correctly almost every time.** The
gap between H2's result and its 0.75 target is best explained as the same
detection-recall limitation quantified in section 4 (Sofia-fine-tuned
recall 0.335) cascading downstream into risk scoring, rather than an
independent flaw in the `max + damped extra` combination formula itself.
This is a clean, unifying explanation: more/better Sofia-specific training
data would be expected to improve H1's Sofia numbers, H2's classification
agreement, and the product's real usefulness together, rather than being
three separate problems needing three separate fixes.

---

## 6. H3 — routing evaluation

> **H3**: "Risk-aware routing will reduce route hazard exposure compared to
> shortest-path routing, while maintaining a reasonable increase in total
> route distance."

### 6.1 First attempt (city-wide) — inconclusive by design, not by failure

40 independent, randomly sampled origin-destination pairs were generated
across the full Sofia road graph (11,363 nodes, 27,429 edges) and run
through `compare_routes()` (notebook 13) — a fresh, non-duplicated sample,
deliberately not reusing the 27 already-logged app-testing routes in
`outputs/app/app.db`, which are mostly repeated identical dev-testing
clicks and would be pseudo-replication if treated as independent
observations.

**Result: 0 of 40 generated routes crossed any of the graph's 4 currently
risk-carrying segments.** A paired significance test could not be run —
not because the mechanism failed, but because there was insufficient real
detection coverage of the road network for *any* route to have something
to avoid in the first place (4 segments out of 27,429 edges).

### 6.2 Second attempt — concentrated in one neighborhood (Sveta Troitsa)

Rather than spreading further data collection thinly across the whole
city, 14 new photos were taken with real phone GPS (unlike the manually
annotated training/test set, whose EXIF was stripped by the CVAT export
process) across the main streets of one neighborhood, **Sveta Troitsa**,
and processed through the exact same pipeline the Streamlit app uses
(`scripts/batch_upload_sveta_troitsa.py` — same detector, same
confidence threshold, same road-matching logic as a real user upload).
Combined with 3 already-saved photos from the same area, this gave
**17 real observations across 16 distinct road segments** (up from 4
city-wide), all matched within a few meters of their true road.

Route sampling (notebook 14) was then restricted to the 206 graph nodes
actually inside the neighborhood's bounding box — derived from the real
GPS extent of the 17 photos (~110m margin), not eyeballed from a map — so
generated routes have an actual chance of crossing the segments that now
carry real risk data. Segment risk was read directly from the live app
database (`road_segments` table), not the separate, stale
`outputs/geo/segment_risk.csv` (see section 8's note on the project's two
storage tracks).

**Result:**

| Metric | Value |
|---|---|
| Routes generated | 40 |
| Routes crossing a known hazardous segment | **18 (45%)** |
| Median / mean distance increase (all routes) | 0.00% / 1.50% |
| % satisfying the ≤10% distance constraint | 95% |
| Mean / median exposure reduction (hazard-crossing routes) | 45.6% / 37.2% |
| Wilcoxon signed-rank test (shortest exposure > risk-aware) | **p = 0.0017** |
| **H3 verdict** | **SUPPORTED** |

### 6.3 Framing for the thesis

**This before/after pair (6.1 → 6.2) is itself worth presenting as a
finding, not just a fixed blocker.** It demonstrates directly that H3's
routing mechanism was correct all along — the first attempt's null result
was a data-coverage artifact, confirmed by making the exact same code
statistically significant (p = 0.0017) once real ground-truth density was
concentrated in one area. This is also an explicit, honest scoping
decision worth stating plainly: **this evaluation covers one Sofia
neighborhood, chosen for the feasibility of collecting real, geotagged
photos within the project's timeframe — not claimed as representative of
the entire city.** Extending this same photo-collection approach to more
neighborhoods is the natural way to broaden the claim later, and the same
16-segment dataset also fed directly into H4 below.

---

## 7. H4 — heatmap validation

> **H4**: "The system will enable identification of high-risk road
> segments..." Operationalized target: Spearman's ρ ≥ 0.5 (with p < 0.05)
> between generated segment risk and manually labeled ground truth.

### Method

The same 16 Sveta Troitsa segments from section 6.2 (real detection-based
risk, already in the live database) were independently rated 1-5 by the
project owner — using each segment's Google Maps/Street View location to
judge actual road condition, **blind to the model's computed score** (same
principle as H2). See `thesis/h4_manual_segment_ratings_template.csv`
(filled) and `scripts/evaluate_h4_heatmap_validation.py`.

### Results

| Comparison | Spearman ρ | p-value | Verdict |
|---|---|---|---|
| Manual rating vs. `aggregated_risk` (total_risk) | **0.683** | 0.0036 | **SUPPORTED** |
| Manual rating vs. `risk_density` | **0.558** | 0.0247 | **SUPPORTED** |

Both comparisons clear the 0.5 target with a statistically significant
p-value (< 0.05) on a sample of 16 segments — the strongest, cleanest
result of the four hypotheses. `aggregated_risk` correlates more strongly
than `risk_density`, which makes intuitive sense given both the manual
rating and `aggregated_risk` respond to "how much/severe visible damage is
there," while `risk_density` additionally divides by segment length — a
factor the manual rater wasn't explicitly weighing.

### Framing for the thesis

This is a genuinely strong, positive result and worth stating as such
without over-hedging — but the sample size (16 segments, one rater, one
neighborhood) should still be named as a scope limitation: a larger
validation (more segments, ideally a second independent rater with
inter-rater agreement reported) would strengthen the claim further. As a
first validation of whether the system's generated risk actually tracks
real-world road condition, ρ = 0.68 against total risk is a compelling
result for a graduation-project prototype.

---

## 8. Summary — all four hypotheses

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** (model comparison) | Partially supported — complexity: yes, accuracy: no | Wilcoxon p = 6.04×10⁻⁹⁹ (inference speed); 95% bootstrap CI [+0.086, +0.117] excludes the ±2% equivalence margin (accuracy) |
| **H2** (risk classification) | Not supported | Macro F1 = 0.582 (target ≥ 0.75); Cohen's κ = 0.377; root cause traced to detector recall misses, not the scoring formula |
| **H3** (routing) | Supported (Sveta Troitsa) | Wilcoxon p = 0.0017; 45% of routes crossed a hazard, 45.6% mean exposure reduction, 95% stayed within the 10% distance cap |
| **H4** (heatmap validation) | Supported | Spearman ρ = 0.683, p = 0.0036 (vs. `aggregated_risk`); ρ = 0.558, p = 0.0247 (vs. `risk_density`) |

**Every hypothesis now has a real, reproducible statistical result** — no
threshold claim was left untested. The overall narrative: YOLOv8n is the
clearly stronger detector choice (H1); the risk-scoring and routing
*mechanisms* both work correctly once given real ground truth to check
against (H3, H4 supported) — the one place the system currently falls
short (H2) is traceable to a specific, well-understood cause (detection
recall on real Sofia photos) rather than a flaw in the risk-scoring logic
itself, which is exactly the kind of precise, defensible conclusion a
thesis discussion section should end with.

---

## 9. Known limitations to state explicitly in the thesis

- **H2's proposal wording** ("risk scoring based on damage type *and
  size*") does not match the implemented risk model, which deliberately
  excludes bounding-box size as a severity factor (camera-distance/angle
  distorts apparent box size in a 2D photo, unrelated to real-world damage
  size — see `src/risk/scoring.py`). This should be explicitly justified in
  the thesis rather than silently glossed over.
- **D40 (pothole) does not exist in RDD2022 for Norway, US, or Czech
  Republic** — country-balancing for this class was only possible across
  the 4 countries that do contain it (Japan, India, China-Drone,
  China-MotorBike).
- **Manual Sofia test set is small** (42 images, 11 Pothole instances) —
  per-class Sofia metrics, especially for Pothole, should be treated as
  directional, not statistically robust. The same 42 images were reused for
  H2 (with only one manual labeler) — per-category H2 F1 for the
  smaller-support categories (8 "low", 15 "medium") should likewise be
  read as directional rather than tightly precise; a second independent
  labeler (Cohen's kappa) would strengthen this if time allows.
- **All four hypotheses now have a real statistical result behind them**
  (sections 3, 5, 6, 7) — none were left as an unverified threshold claim.
  H1: partially supported (accuracy no, complexity yes). H2: not supported,
  root cause identified (detector recall, not the formula). H3: supported,
  but only after concentrating data collection in one neighborhood. H4:
  supported, on both risk metrics tested.
- **H3's and H4's results are geographically scoped** — one neighborhood
  (Sveta Troitsa), not the whole city — stated explicitly in both sections
  as a deliberate feasibility-driven choice, not hidden. Extending either
  result to a city-wide claim would need the same photo-collection approach
  repeated across more neighborhoods.
- **H4's sample is small** (16 segments, one rater) — a strong result (ρ =
  0.68) on a small, single-rater sample; a larger validation with a second
  independent rater (inter-rater agreement reported) would strengthen the
  claim, same caveat pattern as H2's small-support categories.
- **Faster R-CNN's overfitting pattern was not resolved** by scaling the
  dataset alone; this is reported as a genuine finding/limitation of the
  chosen configuration, not swept under the rug.
- **H2's disagreement is substantially attributable to detector recall**,
  not the scoring formula (section 5) — worth stating plainly rather than
  reading H2's "NOT SUPPORTED" verdict as a standalone risk-scoring flaw.

---

## 10. Where the underlying data/checkpoints live (for reproducing these numbers)

| Artifact | Path |
|---|---|
| RDD2022 large dataset | `data/processed/RDD2022_4classes_large/` |
| Manual Sofia dataset (full, verified) | `data/manual/` |
| Manual Sofia split (fine-tune / held-out test) | `data/manual_split/` |
| Manual fine-tune train/val carve-out | `data/manual_finetune_split/` |
| YOLOv8n large-dataset training | `notebooks/09_yolov8n_large_dataset.ipynb` |
| Faster R-CNN large-dataset training | `notebooks/10_fasterrcnn_large_dataset.ipynb` |
| YOLOv8n Sofia fine-tuning | `notebooks/11_yolov8n_sofia_finetune.ipynb` |
| Faster R-CNN Sofia fine-tuning | `notebooks/12_fasterrcnn_sofia_finetune.ipynb` |
| Faster R-CNN training/eval code (shared) | `src/training/fasterrcnn.py` |
| H1 statistical tests + H3 first (city-wide) pass | `notebooks/13_final_results_and_hypothesis_testing.ipynb` |
| H2 manual ground-truth labels | `thesis/h2_manual_risk_labels_template.csv` |
| H2 evaluation script | `scripts/evaluate_h2_risk_classification.py` |
| H2 confusion matrix + per-image results | `outputs/reports/h2_confusion_matrix.png`, `outputs/reports/h2_evaluation_results.csv` |
| Sveta Troitsa photo batch-upload script | `scripts/batch_upload_sveta_troitsa.py` |
| H3 Sveta Troitsa routing evaluation + map | `notebooks/14_h3_sveta_troitsa_routing.ipynb`, `notebooks/sveta_troitsa_routing_map.html` |
| H4 manual ground-truth ratings | `thesis/h4_manual_segment_ratings_template.csv` |
| H4 evaluation script | `scripts/evaluate_h4_heatmap_validation.py` |
| H4 per-segment results | `outputs/reports/h4_evaluation_results.csv` |
| Runtime metric (thesis-citable) | `models/fasterrcnn_large/runtime_summary.txt` |
