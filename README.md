# Road Guard AI

Graduation project (Applied Data Science & AI, Breda University of Applied
Sciences) building an end-to-end pipeline that detects road damage in street
photographs, converts detections into an explainable risk score, matches
that risk to the OpenStreetMap road network, and uses it for risk-aware
route planning and road-maintenance prioritisation.

```
road image → GPS location → damage detection → risk score →
OpenStreetMap road segment → risk map → safer route
```

The full write-up, methodology, and results are in
[`thesis/Thesis_Vladislav_Stoimenov.pdf`](thesis/Thesis_Vladislav_Stoimenov.pdf).
The detailed functional specification the application and evaluation
scripts are built against is
[`Streamlit_application_instructions.md`](Streamlit_application_instructions.md).

## Contents

- [What it does](#what-it-does)
- [Before you start](#before-you-start)
- [Key results](#key-results)
- [Repository structure](#repository-structure)
- [Data used to train the models](#data-used-to-train-the-models)
- [Models and risk scoring](#models-and-risk-scoring)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the application](#running-the-application)
- [Using the application](#using-the-application)
- [Reproducing the analysis and experiments](#reproducing-the-analysis-and-experiments)
- [Limitations](#limitations)
- [Data licensing and attribution](#data-licensing-and-attribution)
- [Author](#author)

## What it does

- **Detection:** YOLOv8n and Faster R-CNN, trained on a class/country-balanced
  subset of [RDD2022](https://github.com/sekilab/RoadDamageDetector) and
  fine-tuned on manually collected photographs from Sofia, Bulgaria, detect
  four damage types: longitudinal crack, transverse crack, alligator crack,
  and pothole.
- **Risk scoring:** each accepted detection gets a rule-based risk score
  from its damage class and confidence (`src/risk/`), combined per image and
  per road segment into a low/medium/high category.
- **Geospatial matching and routing:** detections are matched to the nearest
  OpenStreetMap road edge (`src/geospatial/`) and used to bias a
  shortest-path routing algorithm away from known hazards (`src/routing/`).
- **Streamlit application:** a four-page app (`app.py`, `pages/`) — Dashboard,
  Upload and Detect, Damage Map, and Safer Route Planner — that runs the full
  pipeline interactively and stores results in a local SQLite database.

## Before you start

This repository ships the one model the app actually needs, but not the
raw RDD2022 dataset or the research-comparison model checkpoints — those
are large and git-ignored (`*.pt`, `notebooks/runs/`, `data/raw/`). To
actually run the app or the training notebooks on a fresh machine, be
aware of the following before following the steps below:

- **The deployed model is included; the comparison models are not.** The
  fine-tuned YOLOv8n checkpoint the app uses by default
  (`models/yolov8n_sofia_finetune/best.pt`, ~6 MB) is committed to this
  repository, so the app works out of the box with no training required.
  The Faster R-CNN checkpoints under `models/` (the H1 architecture
  comparison, ~170 MB each) are **not** included and are git-ignored — the
  app doesn't use them, only `notebooks/13_final_results_and_hypothesis_testing.ipynb`
  does. If you need those, you either need to train them yourself (see
  [Reproducing the analysis](#reproducing-the-analysis-and-experiments) —
  realistically several hours on a CUDA GPU) or obtain a copy from the
  author.
- **No raw dataset included, and a known gap in reproducing it.**
  `data/raw/` is expected to contain RDD2022 already converted into a
  `{train,val,test}/{images,labels}/*.txt` YOLO-label layout — that is
  **not** the format the official RDD2022 release ships in (it ships as
  per-country Pascal VOC XML annotations). This repository does not
  currently contain the script that performed that original XML → YOLO
  conversion, so downloading RDD2022 and running
  `scripts/create_rdd2022_4classes_*.py` directly against it will not work
  out of the box. `src/data/inspect_rdd2022.py` can inspect a raw
  XML-format download, but is not a converter. If you need to reproduce
  training from scratch, you will need to either locate that original
  conversion step or write a new one that outputs the label format
  described in [Data used to train the models](#data-used-to-train-the-models).
- **Disk space:** the full `data/` directory (raw RDD2022 plus all
  processed/manual datasets) is roughly 13 GB; trained model checkpoints
  under `models/` another ~1.3 GB; training-run outputs accumulate further
  space under `notebooks/runs/` as you train.
- **Hardware:** training was done on a CUDA-capable GPU; Faster R-CNN in
  particular took multiple hours per run even on one. CPU-only training is
  not practical. Whether the Streamlit app's inference step works
  acceptably on CPU alone has not been verified — see
  [Installation](#installation).

None of this blocks exploring the code, the thesis, or the evaluation
outputs already committed under `outputs/` and `thesis/`.

## Key results

Summarised from the thesis (see the PDF for full methodology, statistics,
and discussion):

| Hypothesis | Question | Result |
|---|---|---|
| H1 | YOLOv8n vs. Faster R-CNN accuracy/complexity | YOLOv8n reached mAP50-95 0.300 vs. 0.186 (large dataset) and ran far faster (median 22 ms vs. 307 ms/image) — partially supported: the complexity prediction held, the predicted near-parity in accuracy did not |
| H2 | Can the rule-based risk score classify risk correctly? | Macro F1 0.582 against independently labelled Sofia images — not supported at the 0.75 target; misses traced to detector recall, not the scoring rule |
| H3 | Does risk-aware routing reduce hazard exposure? | Median exposure reduction 37% (mean 46%) with 95% of routes within a 10% distance increase (Wilcoxon p = 0.0017) — supported |
| H4 | Does generated segment risk match manually rated risk? | Spearman's ρ = 0.683 (p = 0.0036) against independently rated road segments — supported |

Domain adaptation: both detectors, trained only on RDD2022, perform far
worse on real Sofia photographs (YOLOv8n mAP50-95 drops from 0.300 to
0.041). Fine-tuning on 131 local images recovers much of that gap
(mAP50-95 0.148, F1 0.189 → 0.347).

## Repository structure

```
.
├── app.py                    # Streamlit entry point (landing page)
├── pages/                    # The app's other three pages (see "Using the application")
├── src/                      # Pipeline code (see table below)
├── scripts/                  # Command-line scripts: dataset build, splitting, evaluation
├── notebooks/                # Numbered notebooks: training and evaluation, run in order
├── config/                   # Application and risk-scoring configuration (YAML)
├── configs/datasets/         # YOLO-format dataset config used by the training notebooks
├── data/                     # Datasets (raw, processed, manually collected, uploaded)
├── models/                   # Deployed YOLOv8n checkpoint (committed) + Faster R-CNN checkpoints (git-ignored)
├── outputs/                  # App database, routing/segment-risk results, evaluation reports
├── thesis/                   # Thesis source, figures, analysis notes, evidencing documents
├── requirements.txt
└── Streamlit_application_instructions.md   # Functional spec the app/scripts are built against
```

`src/` in more detail:

| Package | Contents |
|---|---|
| `src/detection/` | Detector wrapper (`detector.py`) and annotated-image drawing (`annotate.py`) |
| `src/risk/` | Per-detection scoring (`scoring.py`), the shared max+damped-rest combinator (`combine.py`), image-level (`observation_risk.py`) and road-segment-level (`segment_aggregation.py`) aggregation |
| `src/geospatial/` | Sofia road-graph loading/caching (`network.py`), GPS-to-edge matching (`matching.py`), address geocoding (`geocoding.py`), segment risk-filling (`fill_segments.py`) |
| `src/routing/` | Shortest-path vs. risk-aware routing cost and comparison (`router.py`) |
| `src/training/` | Faster R-CNN training/evaluation loop (`fasterrcnn.py`; YOLOv8n uses the Ultralytics CLI/API directly from the notebooks) |
| `src/data/` | Dataset/record helpers: structured detection records, GPS utilities, the torchvision-style YOLO dataset loader |
| `src/app/` | Streamlit-specific code: SQLite access (`database.py`), the Upload & Detect pipeline (`upload_pipeline.py`), map building (`map_utils.py`), sidebar (`sidebar.py`), theming (`theme.py`), cached resource loading (`resources.py`) |
| `src/config.py` | Loads `config/app_config.yaml` and `config/risk_weights.yaml` |

## Data used to train the models

**RDD2022** ([Arya et al., 2024](https://doi.org/10.1002/gdj3.215)) is the
primary training dataset: 26,869 training, 5,758 validation, and 5,758 test
images from seven country/collection sources (Japan, India, the Czech
Republic, Norway, the United States, and two Chinese collection methods).
Only four of its damage classes are used — D00 (longitudinal crack), D10
(transverse crack), D20 (alligator crack), D40 (pothole) — a fifth,
unofficial class is filtered out.

Three processed training sets were built at increasing scale (see
`scripts/create_rdd2022_4classes_*.py`):

| Dataset | Train | Val | Test | Notes |
|---|--:|--:|--:|---|
| Small | 277 | 60 | 60 | Used to validate the pipeline end to end before scaling up |
| Medium | 1,034 | 224 | 221 | Revealed class/country imbalance (D00 over-represented; ~46% Japan) |
| Large | 4,291 | 597 | 592 | Explicitly balanced per class and country (water-filling allocation); used for all reported results |

**Manually collected Sofia data** (`data/manual/`, 201 photographs,
collected and annotated by the author in CVAT) is used for
domain-adaptation fine-tuning, split into:

| Split | Images | Purpose |
|---|--:|---|
| Fine-tune train | 131 | Fine-tuning both detectors from their RDD2022-large weights |
| Fine-tune validation | 28 | Monitoring for overfitting during fine-tuning |
| Held-out test | 42 | Before/after fine-tuning comparison; also the H2 evaluation sample |

A second, separate set of 17 geotagged photographs from the Sveta Troitsa
neighbourhood (`data/pipeline_test/`, processed via
`scripts/batch_upload_sveta_troitsa.py`) supplies the real segment-risk
data used for the H3 (routing) and H4 (heatmap validation) evaluations.

RDD2022 itself is not included in this repository (download separately;
see [Data licensing and attribution](#data-licensing-and-attribution)) and
is never modified by any script.

## Models and risk scoring

- **YOLOv8n** (Ultralytics) and **Faster R-CNN** (ResNet50-FPN,
  torchvision) are trained identically at each dataset scale for a fair
  comparison, then both fine-tuned on the manual Sofia data.
- The deployed app uses the fine-tuned YOLOv8n checkpoint
  (`config/app_config.yaml: model.path`), chosen for its large real-world
  (Sofia) accuracy advantage and much faster inference, not because it
  scored higher on RDD2022 alone.
- **Risk score** per detection: `base_severity[class] × confidence_multiplier(confidence)`,
  with base severities 1.0 (longitudinal crack) to 3.0 (pothole) and a
  confidence multiplier narrowed to `[0.7, 1.0]` (`config/risk_weights.yaml`).
  Multiple detections combine as `max(risks) + 0.3 × sum(the rest)`, so the
  single worst hazard dominates while co-occurring damage still counts.
- **Routing cost** per road edge: `length_m + alpha × risk_density`, with
  `alpha = 15000` found empirically against the Sofia pilot data.

See the thesis Methodology chapter for the full derivation and the design
criteria behind these choices.

## Installation

Requires **Python 3.10**.

`requirements.txt` was generated from the project's own environment and
pins a CUDA 12.8 build of PyTorch (`torch==2.11.0+cu128`,
`torchvision==0.26.0+cu128`, `torchaudio==2.11.0+cu128`). These specific
`+cu128` wheels are hosted on PyTorch's own package index, not the default
PyPI index, so running `pip install -r requirements.txt` as a single plain
command will fail to resolve them on most machines — **install PyTorch
first, separately, then the rest:**

```bash
# 1. Install PyTorch matching your hardware first.
#    - NVIDIA GPU with CUDA 12.8 (matches this project's pinned versions):
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
#    - Different CUDA version or CPU-only: pick the right command for your
#      system from https://pytorch.org/get-started/locally/ instead. Using
#      a different torch/torchvision version than the ones above has not
#      been tested against this codebase.

# 2. Then install the rest of the dependencies.
pip install -r requirements.txt
```

CPU-only inference/training has not been verified to work with this
codebase — training in particular was only ever run on a CUDA GPU (see
[Before you start](#before-you-start)).

An internet connection is needed the first time the app or notebooks fetch
the Sofia road network (via OSMnx) or geocode an address.

## Configuration

All tunable values live in two YAML files under `config/`, loaded via
`src/config.py`:

- **`config/app_config.yaml`** — active model path and version label,
  confidence thresholds, the geospatial pilot area (Sofia, Bulgaria, 8 km
  radius), the damage-class display names, the maximum GPS-to-road
  matching distance (50 m), and application file paths (database, uploads,
  exports).
- **`config/risk_weights.yaml`** — the risk-scoring formula's parameters:
  base severity per class, confidence floor/threshold, the combination
  damping factor `beta`, the `[0,1]` normalisation ceiling, the
  low/medium/high category thresholds, and the routing weight `alpha`.

Change a value and restart the app to pick it up (values are cached per
process via `functools.lru_cache`).

## Running the application

```bash
streamlit run app.py
```

The app expects:

- a trained detector at the path set in `config/app_config.yaml` under
  `model.path` — the fine-tuned YOLOv8n checkpoint it defaults to,
  `models/yolov8n_sofia_finetune/best.pt`, **is included in this
  repository** (it's small, ~6MB), so the app works out of the box without
  retraining anything. (The larger Faster R-CNN checkpoints under `models/`
  are a research comparison artifact, not used by the app, and are
  git-ignored — see [Before you start](#before-you-start));
- the cached Sofia road network at `outputs/geo/sofia_drive_network.graphml`
  (downloaded automatically via OSMnx on first use if missing);
- a writable location for the SQLite database and uploaded images.

**No database setup step is needed, and you don't start from empty.**
This repository includes the actual SQLite database
(`outputs/app/app.db`, set by `config/app_config.yaml: paths.database`)
and uploaded images (`data/uploads/`) behind the results shown in the
thesis — 19 observations, 42 detections, and 17 risk-scored road segments
in the Sveta Troitsa pilot area, matching the Appendix D screenshots. Open
the app and the Dashboard, Damage Map, and Safer Route Planner already
have this data to show. If the database or an uploads directory is ever
missing, it's recreated automatically from scratch instead: every page
calls `init_db()` on load, which creates the containing directory if
needed and runs `CREATE TABLE IF NOT EXISTS` for the schema (four tables:
`observations`, `detections`, `road_segments`, `routes`) — there's no
migration tool or separate init command either way.

## Using the application

The app has four pages, listed in the sidebar (Streamlit's automatic
multipage navigation from `pages/`). Every page shares the same sidebar:
the active model's name and availability, how many images/detections are
stored so far, a **confidence threshold** slider (detections below this
are discarded before risk scoring), and a button to clear active filters.

1. **Dashboard** (`app.py` is the landing page; `pages/1_Dashboard.py`) —
   summary cards (images, detections, high-risk segments, model status),
   detections-per-class and observations-per-risk-category breakdowns, a
   table of recent uploads, and a small overview map. The starting point
   once you have some data saved.
2. **Upload & Detect** (`pages/2_Upload_and_Detect.py`) — upload one or
   more road photographs (batch supported, duplicates skipped by content
   hash). Location is read from EXIF GPS if present, otherwise it can be
   entered manually. The detector runs automatically; review the annotated
   image and each detection's class, confidence, and risk contribution,
   then confirm to save the observation (nothing is written to the
   database until you confirm).
3. **Damage Map** (`pages/3_Damage_Map.py`) — every confirmed observation
   as a marker, plus the road segments it has affected, coloured by
   aggregated risk, with an optional heatmap layer and filters (filters
   only affect what's displayed, not what's stored or used for routing).
4. **Safer Route Planner** (`pages/4_Safer_Route_Planner.py`) — pick an
   origin and destination (or geocode an address) within the Sofia pilot
   area to compare the shortest route against a risk-aware alternative:
   both are drawn on the map alongside their distance and hazard-exposure
   difference.

The database already included with this repository (see
[Running the application](#running-the-application)) has the same 19
observations shown in the thesis, so the Dashboard, Damage Map, and route
comparisons all have real data from the first run — try **Upload & Detect**
to add more on top of it. `scripts/batch_upload_sveta_troitsa.py` is what
originally bulk-populated it from the Sveta Troitsa test photographs in
`data/pipeline_test/`, and can be re-run safely (it skips anything already
saved, by content hash).

## Reproducing the analysis and experiments

Dataset construction and training are organised as numbered notebooks
under `notebooks/`, intended to be run in order:

1. `01_explore_rdd2022.ipynb`, `02_visual_inspection_rd2022.ipynb` —
   dataset exploration.
2. `03_first_yolo_training_sample_dataset.ipynb`,
   `05_yolov8n_retrain_medium_dataset.ipynb`,
   `09_yolov8n_large_dataset.ipynb` — YOLOv8n training at each dataset
   scale.
3. `07_fasterrcnn_small_dataset.ipynb`, `08_fasterrcnn_medium_dataset.ipynb`,
   `10_fasterrcnn_large_dataset.ipynb` — Faster R-CNN training at each
   scale.
4. `11_yolov8n_sofia_finetune.ipynb`, `12_fasterrcnn_sofia_finetune.ipynb`
   — domain-adaptation fine-tuning on the manual Sofia dataset.
5. `13_final_results_and_hypothesis_testing.ipynb` — H1/H2 statistical
   testing on the large-dataset and Sofia results.
6. `14_h3_sveta_troitsa_routing.ipynb` — H3 risk-aware routing evaluation.

> **Note:** the commands below assume `data/raw/RDD2022/` already exists in
> the converted `{train,val,test}/{images,labels}/*.txt` layout these
> scripts read. See [Before you start](#before-you-start) — reproducing
> this from a fresh RDD2022 download currently requires a conversion step
> that is not included in this repository.

The processed RDD2022 datasets these notebooks train on are built by:

```bash
python scripts/create_rdd2022_4classes_small.py
python scripts/create_rdd2022_4classes_medium.py
python scripts/create_rdd2022_4classes_large.py
```

and the manual Sofia dataset is verified and split by:

```bash
python scripts/verify_manual_dataset.py
python scripts/split_manual_dataset.py
python scripts/split_manual_finetune_val.py
```

H2 and H4 (risk classification and heatmap-validation hypotheses) are
evaluated against manually completed ground-truth templates in `thesis/`:

```bash
python scripts/evaluate_h2_risk_classification.py
python scripts/evaluate_h4_heatmap_validation.py
```

The qualitative usability survey is analysed by:

```bash
python scripts/analyze_qualitative_survey.py
```

`scripts/batch_upload_sveta_troitsa.py` replays a batch of photographs
through the same pipeline the app's Upload & Detect page uses, to build
the Sveta Troitsa segment-risk data the H3/H4 evaluations and the Safer
Route Planner demo rely on.

Dataset construction uses a fixed random seed for reproducibility; see
each script's module docstring for its specific sampling method.
`data/raw/`, `data/processed/`, and the training-run output directories
(`notebooks/runs/`) are git-ignored: they are either large third-party
data or deterministically regenerated by the commands above.

## Limitations

Briefly (see the thesis Discussion and Ethical Considerations chapters for
the full treatment):

- Detection recall on real Sofia photographs is the main bottleneck — it
  is the traced cause of H2 not reaching its target, more than the risk
  formula itself.
- H3 and H4's supported results are geographically scoped to one
  neighbourhood (Sveta Troitsa), chosen for data-collection feasibility,
  and are not claimed to generalise across all of Sofia without further
  data collection.
- H2's ground truth and H4's segment ratings were each assigned by a
  single rater.
- The app does not anonymise faces, licence plates, or other identifiable
  content in uploaded images, and has no access-control or data-retention
  policy — it is a local, single-user research prototype, not something
  deployed for public use.

## Data licensing and attribution

This project's own code is licensed under the [MIT License](LICENSE).

RDD2022 is used under the Creative Commons Attribution 4.0 International
licence (CC BY 4.0); see
[Arya et al., 2024](https://doi.org/10.1002/gdj3.215) and the
[RDD2022 dataset release](https://doi.org/10.6084/m9.figshare.21431547.v1).
It is not redistributed in this repository. The manually collected Sofia
photographs and the thesis document itself are the author's own work,
produced specifically for this project, and are not covered by the MIT
licence above.

## Author

Vladislav Stoimenov — Applied Data Science & AI, Breda University of
Applied Sciences. Supervisor: Karna Rewatkar.
