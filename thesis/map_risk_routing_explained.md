# The Map, Risk Scoring, and Routing — How It Works and Why

This document explains, in plain terms, everything behind the Damage Map and Safer
Route Planner pages of the Streamlit app: how a photo becomes a point on a map, how
that point gets a risk number, how road segments get colored, and how the "safer
route" is actually calculated. It's meant to be enough to explain the logic to a
mentor without needing to read the code.

Source of truth: every number quoted here lives in
[config/risk_weights.yaml](../config/risk_weights.yaml) and
[config/app_config.yaml](../config/app_config.yaml) — nothing is hardcoded in the
modules, so tuning a value means editing the config, not the code.

---

## 1. The pipeline, end to end

```
photo uploaded
   -> GPS read from EXIF (or entered manually)
   -> YOLOv8n detects damage (boxes + class + confidence)
   -> each detection gets a risk score
   -> the image's detections combine into one observation risk score
   -> the point is snapped to the nearest road segment (OpenStreetMap edge)
   -> the affected road segment's aggregated risk is recomputed
   -> map shows the point, the road-segment coloring, and a heatmap
   -> route planner uses per-segment risk to penalize a route through them
```

Every stage below corresponds to one module in `src/`.

---

## 2. Turning a photo into a located, detected observation

### 2.1 Getting a location — `src/data/gps_utils.py`

Phone photos usually carry GPS coordinates inside their EXIF metadata (the same
hidden data that stores camera settings). This module reads that block and converts
it from the raw degrees/minutes/seconds format into a plain decimal latitude/longitude.

- If EXIF has no GPS (common for stripped/research datasets like RDD2022), the
  function returns `None` — this is expected, not an error. The upload page then
  asks the user to enter coordinates manually. `coordinate_source` on every saved
  observation records which of the two happened (`"EXIF"` or `"manual"`), so it's
  always traceable later.

### 2.2 Detecting damage — `src/detection/detector.py`

A thin wrapper around the trained YOLOv8n model (currently the RDD2022-large
model fine-tuned on real Sofia photos, `yolov8n_sofia_finetune` — see
`thesis/model_training_results_and_discussion.md` section 4 for why the
fine-tuned model was chosen over the stronger-on-paper RDD2022-only model).
It runs inference and returns a clean Python list of `Detection` objects
(class, confidence, bounding box) instead of exposing raw Ultralytics result
objects to the rest of the app.

One deliberate design choice here: the model's raw training class names (e.g.
`D00_Longitudinal_Crack`, the official RDD2022 code) are translated to a
professional display name (`Longitudinal Crack`) **at this one point**, via
`config/app_config.yaml`'s `damage_class_display_names`. Every downstream
consumer — risk scoring, map popups, database, exports — only ever sees the clean
name, so there's no risk of the two naming schemes drifting apart across modules.

Only detections at or above the app's confidence threshold (default 0.25,
adjustable between 0.10 and 0.90) are returned at all — a detection below that never
enters the risk/mapping pipeline in the first place.

---

## 3. Risk scoring — how a detection becomes a number

This is the part most worth being able to explain clearly, since it's the core
"decision logic" of the project. It happens in three layers, each building on the
one below it.

### 3.1 Layer 1 — per-detection score (`src/risk/scoring.py`)

```
detection_risk = base_severity[damage_class] × confidence_multiplier(confidence)
```

**Base severity** is a fixed number per damage class, representing how dangerous
that *type* of damage inherently is — not how big it looks in the photo:

| Damage class | Base severity |
|---|---|
| Longitudinal Crack | 1.0 |
| Transverse Crack | 1.2 |
| Alligator Crack | 2.0 |
| Pothole | 3.0 |

**Why not use bounding-box size as a severity multiplier?** This was a deliberate
decision. A box's size in a 2D photo depends on how close the camera was and the
angle it was taken from — not on the real-world size of the damage. A phone held
low and close would make a small crack look huge. Using box area would let *camera
technique* distort the risk score, so it's intentionally excluded, even though an
earlier version of the spec suggested it.

**Confidence multiplier** — confidence is used as a *mild* nudge, not a dramatic
swing. The reasoning: confidence already gated whether a detection is even
considered ("above the 0.25 threshold or discard it" happens earlier, in the
detector). Once a detection has cleared that bar, a 0.9-confidence pothole isn't
*physically* three times as dangerous as a 0.3-confidence one — we're just less
sure the low-confidence one is real. So instead of confidence scaling risk linearly
from 0 to 1, it's remapped onto a narrow band:

```
confidence <= 0.25 (the threshold)  -> multiplier = 0.7 (the floor)
confidence == 1.0                    -> multiplier = 1.0
in between                           -> linearly interpolated
```

So the multiplier only ever softens a score by at most 30%, never crushes it.

### 3.2 Layer 2 — combining multiple detections (`src/risk/combine.py`)

A single photo, or a single road segment, often has more than one detected damage
instance. They need to combine into one number. The formula used everywhere in this
project is:

```
combined_risk = max(all detection risks) + β × sum(all the OTHER detection risks)
```

with `β = 0.3` (beta, the damping factor).

**Why this shape, instead of just summing or averaging?**
- A plain **sum** would mean five faint hairline cracks outscore one real pothole —
  wrong, since one pothole is the more urgent hazard.
- A plain **average** would mean one pothole photographed alongside a few faint
  cracks gets *diluted* — also wrong, for the same reason.
- **max + damped extra** fixes both: the single worst hazard always dominates the
  score (so a pothole "wins" the assessment), but additional co-occurring damage
  still adds something to the total rather than being ignored outright — a segment
  with a pothole *and* three cracks is treated as slightly worse than one with just
  the pothole.

This exact formula is shared code (`combine_risks()`), used identically at two
different levels so the same reasoning applies everywhere:
- **Image level** (`src/risk/observation_risk.py`) — combining one photo's
  detections into one observation score.
- **Segment level** (`src/risk/segment_aggregation.py`) — combining every
  detection ever matched to the same road edge.

### 3.3 Layer 3a — the observation (image) score

```
normalized_score = min(combined_risk / 3.0, 1.0)     # 3.0 = observation_risk_max
category = "low"    if normalized_score < 0.34
           "medium"  if normalized_score < 0.67
           "high"    otherwise
```

The ceiling (3.0) is chosen so that one full-confidence pothole alone already
saturates the score to 1.0 — the worst single-class hazard should already read as
"maximum risk" on its own, without needing multiple detections to get there.

The low/medium/high cutoffs are an even three-way split (thirds) of the 0–1 range —
a deliberately simple, explainable starting point rather than a tuned threshold.
The code comments flag this as something to revisit once real evaluation/labeled
data exists to calibrate against.

If a photo has *no* detections above the confidence threshold, there is no score at
all (not "risk = 0") — it's recorded as a valid "no damage found" result, not
treated as evidence of absence in a mathematical sense.

### 3.4 Layer 3b — the road-segment score

Every detection that gets geo-matched to a road (see Section 4) contributes its
per-detection risk score to that segment's running total. When a segment is
recomputed, the same `max + damped extra` formula combines *all* detections ever
matched to it, producing two numbers:

- **`total_risk`** (a.k.a. `aggregated_risk`) — the raw combined score, unnormalized.
  Used for the **maintenance-priority list**, because how much repair effort a
  segment needs scales with total damage present, not damage "per meter".
- **`risk_density = total_risk / segment_length_meters`** — the same score, divided
  by the real-world length of the road edge (from OpenStreetMap data). Used for
  **routing** (Section 5), specifically so a very long road with several detections
  isn't penalized *more* just because it's long and had more chances to be
  photographed — the penalty reflects damage concentration, not road length.

A segment with zero matched detections has `risk_density = 0` — mechanically, "no
data" and "no damage" look identical to the router. This is stated explicitly in the
code as a known limitation, not hidden: absence of a detected problem is not the
same as verified safety, it just means nothing has been reported there yet.

---

## 4. Matching a point to a road — `src/geospatial/matching.py` and `network.py`

The map and router both need to know *which road* a GPS point belongs to, not just
its raw coordinates. This uses [OSMnx](https://osmnx.readthedocs.io/), a Python
library that downloads real road network graphs from OpenStreetMap.

- **Scope decision:** rather than the whole country, the project targets one real
  city — **Sofia, Bulgaria** — as a pilot area (8km radius around the city center).
  This was a deliberate scope choice: the product is framed as a *municipality*
  tool, a country-wide graph would be hundreds of thousands of edges and slow to
  route over, and Sofia already contains one of the project's two real geotagged
  test photos.
- The road graph is downloaded once and **cached to disk** (`sofia_drive_network.graphml`)
  so the app and pipeline don't re-hit the OpenStreetMap API on every run.
- Every road in the graph is an "edge" identified by a triple `(u, v, key)` — two
  endpoint node IDs plus a disambiguation key for parallel roads between the same
  two intersections. This project encodes that triple as a single string
  `"u_v_key"` and uses it as the `road_segment_id` everywhere (database, CSV
  exports, the router).
- **Matching** = snapping a GPS point to the nearest edge in the graph
  (`ox.distance.nearest_edges`). A distance check (converting OSMnx's raw
  degree-based distance to an approximate meter distance) flags points that are
  *too far* from any known road (default cutoff: 50m) for manual review, instead of
  silently assigning a photo to a probably-wrong street.
- Points **outside** the Sofia graph's bounding box (e.g. a test photo taken 140km
  away) are deliberately left unmatched (`road_segment_id = ""`) rather than
  force-matched to the nearest — but meaningless — Sofia edge.

---

## 5. The map — `src/app/map_utils.py`

Built with [Folium](https://python-visualization.github.io/folium/) (a Python
wrapper around the Leaflet.js mapping library). The map has three independently
toggleable layers:

1. **Road-segment risk layer** — every road edge that has at least one matched
   detection is drawn as a colored line (green/orange/red, matching its low/medium/
   high category), with a tooltip showing its aggregated risk score, category, and
   how many observations contributed to it.
2. **Observation markers** — one pin per saved photo, colored by that photo's own
   risk category (gray if no damage was found). Clicking a pin opens a popup
   showing the annotated photo (with the model's bounding boxes drawn on it), when
   it was processed, its coordinates and how they were obtained (EXIF vs. manual),
   which damage types were found, the highest confidence, the computed risk score,
   and which road segment it was matched to.
3. **Risk heatmap** — a density/intensity overlay, weighted by each observation's
   risk score (not just point density), giving a quick "where is it bad" visual
   independent of the discrete pins. Off by default (togglable), since it can be
   visually noisy alongside the other two layers.

**Filters** (damage class, risk category, minimum confidence, whether to show
"no damage" photos) only affect what's *displayed* on this page — they never change
what's stored in the database or what the route planner uses. This separation is
intentional: filtering is a viewing convenience, not a data-editing action.

---

## 6. Routing — `src/routing/router.py`

This is where the risk numbers become actionable: comparing a normal shortest
route against a route that tries to avoid known damage.

### 6.1 The cost formula

Every road edge in the graph gets a second "cost" value, on top of its real
physical length:

```
risk_cost = length_meters + alpha × risk_density
```

- The **shortest route** is computed the normal way — shortest path by
  `length_meters` alone (via NetworkX's Dijkstra-based `shortest_path`).
- The **risk-aware route** is the shortest path by `risk_cost` instead — so a
  road segment with high recorded damage effectively looks "longer" to the
  routing algorithm, making the algorithm prefer to route around it *if* the
  detour isn't too costly in real distance.

### 6.2 Why `alpha = 15000`

`alpha` controls how strongly damage is penalized relative to distance. It was
picked **empirically**, by testing it against the real Sofia pilot data collected
during development: below roughly 15,000, the risk-aware route still cut straight
through the single worst known segment, because the risk penalty didn't outweigh
the extra distance of any available detour. At around 15,000, avoiding it finally
became "worth it" in the shortest-path calculation.

This is explicitly **a rule-based prototype value, not a calibrated safety
threshold** — it doesn't correspond to a real unit like "meters per risk point" in
any validated sense, it was tuned by observing behavior on the available test data.
That's exactly why it's exposed as a slider (0–40,000) in the Safer Route Planner
page rather than hardcoded — the mentor or a future user can see directly how
route choice changes as alpha increases.

### 6.3 What gets reported for each route

For both the shortest and risk-aware paths, the router reports:
- **Distance** (meters/km)
- **Hazard segments encountered** — how many edges on the route have any recorded
  detections at all
- **Total risk encountered** — the sum of `total_risk` (not density) across every
  hazardous segment on that specific route, i.e. "how much cumulative damage
  exposure does this route pass through"

The comparison view then derives:
- **Distance difference** — how much longer (or shorter) the safer route is
- **Risk reduction** — how much less total risk exposure the safer route has,
  shown as a percentage

### 6.4 Known scope limitation (documented, not accidental)

The Streamlit app's original spec described a more elaborate routing approach:
generating multiple candidate routes and enforcing a **hard cap** (e.g. never let
the safer route be more than 10% longer). That was explicitly deferred — the
current router uses the simpler single-`alpha` additive-cost approach instead. This
means there's currently no hard guarantee on how much extra distance the "safer"
route can add; `alpha` is the only lever. This is stated openly in the code and in
the app UI itself (a caption on the Safer Route Planner page), not hidden.

---

## 7. What happens when a route or observation is saved

- **Observations** (`src/app/upload_pipeline.py`, `save_observation()`): once a
  user confirms a processed photo, its observation record, all its detections, and
  — if it matched a road segment — that segment's *recomputed* aggregate risk are
  all written to the SQLite database (`outputs/app/app.db`) in one step. Only
  confirmed/saved observations ever affect the map or router; a preview/detection
  test run that isn't saved has no side effects.
- **Routes**: every time "Calculate routes" is used, the comparison (both routes'
  distances, risk exposure, the alpha used, and the resulting percentages) is
  logged to a `routes` table — useful for showing a mentor a history of route
  comparisons actually run during testing/demoing.

(See Section 8 for what these tables look like and why SQLite was chosen for them.)

---

## 8. Data storage — what's stored where, and why

There are actually **two separate storage tracks** in this project, built at
different stages for different purposes. Knowing both — and why they exist
side by side — matters for explaining the project's evolution honestly.

### 8.1 Track 1 — the early standalone pipeline (plain CSV files)

Before the Streamlit app existed, the detection → risk → geospatial pipeline was
built and tested as standalone scripts, each writing its output to a plain CSV or
JSON file on disk:

| File | Written by | Contents |
|---|---|---|
| `outputs/detections/pipeline_test_detections.csv` | `src/data/detection_records.py` | One row per detection: image, GPS, damage type, confidence, bbox, per-detection risk score, and a blank `road_segment_id` waiting to be filled |
| (same file, updated) | `src/geospatial/fill_segments.py` | Fills in `road_segment_id` for every row whose GPS point falls inside the Sofia pilot area |
| `outputs/geo/segment_risk.csv` | `src/risk/segment_aggregation.py` | One row per road segment: length, detection count, `total_risk`, `risk_density` |
| `outputs/geo/route_comparison.json` | `src/routing/router.py` | One shortest-vs-risk-aware route comparison result |
| `outputs/geo/sofia_drive_network.graphml` | `src/geospatial/network.py` | The cached OpenStreetMap road graph itself (see 8.3) |

**Why CSV/JSON for this stage:** while the pipeline logic itself was still being
built and validated stage by stage, the priority was being able to *open a file
and read it* — no database viewer needed, easy to eyeball in Excel or a text
editor, easy to diff between runs, and each script could be run and re-run
independently via its own `if __name__ == "__main__":` block without needing any
shared server or schema migration. This matches the project's early-stage
priority (see Section 4.1/5.1 project notes): prove each pipeline stage works
correctly in isolation before wiring it into a real application.

The limitation that eventually mattered: a flat CSV has no relational structure.
Recomputing "all detections on this one road segment" from `segment_risk.csv`
means re-reading and re-grouping the entire detections CSV every time — fine for
a one-off batch script, increasingly awkward once the app needed to do this
*live*, every time a user uploads a photo, without re-scanning everything from
scratch.

### 8.2 Track 2 — the live Streamlit app database (SQLite)

Once the project moved to an interactive app, storage moved to a proper embedded
relational database: **SQLite**, via `src/app/database.py`, stored as a single
file at `outputs/app/app.db`.

**Why SQLite specifically, and not something heavier (Postgres/MySQL) or
something lighter (staying with CSV):**
- **No server to install or run.** SQLite is just a file — it fits a local,
  single-user graduation-project prototype perfectly; there's no deployment or
  connection-string overhead a marker would need to reproduce the demo.
- **Relational queries where they actually matter.** The app needs things like
  "every detection whose observation matched edge (u, v, key)" — a SQL `JOIN`
  (`get_detection_risks_for_edge`) does this in one query; doing the same thing
  against flat CSVs would mean loading and cross-referencing multiple files by
  hand every time a page loads.
- **Upsert semantics for road segments.** A road segment's aggregated risk needs
  to be *recomputed in place* every time a new photo affects it — not appended
  as a new row. SQLite's `INSERT ... ON CONFLICT DO UPDATE` (`upsert_road_segment`)
  does this atomically, keyed on the same `(edge_u, edge_v, edge_key)` triple used
  as the segment ID everywhere else in the project (Section 4).
- **This still matches the original project scope.** The project's own working
  notes explicitly allowed "CSV, GeoJSON, SQLite, or another simple local
  format" for the prototype, explicitly *not* requiring a full production
  database — SQLite is the most structured option inside that same "keep it
  simple" bound, chosen once the relational need above appeared.

**Four tables, matching the four record types described in Sections 2–6:**

| Table | One row per | Key columns worth noting |
|---|---|---|
| `observations` | one confirmed, saved photo | `file_hash` (unique — see 8.4), `coordinate_source`, `image_risk_score` + `risk_category` (Section 3.3), `matched_edge_u/v/key` (Section 4) |
| `detections` | one accepted detection inside an observation | `image_id` references `observations` (foreign key), `risk_contribution` (Section 3.1); `relative_bbox_area` is stored for display only — explicitly *not* used in the risk formula |
| `road_segments` | one road edge that has ever had a matched detection | primary key is the composite `(edge_u, edge_v, edge_key)` — the same segment-ID scheme as everywhere else; holds both `aggregated_risk` (total) and `risk_density` (Section 3.4) |
| `routes` | one calculated shortest-vs-risk-aware comparison | logs `alpha_used` alongside the result, so a later comparison of two logged routes is still meaningful even if `alpha` was changed between them |

The `detections.image_id -> observations.image_id` foreign key (with
`PRAGMA foreign_keys = ON` enabled on every connection) enforces that a
detection can never exist without its parent observation — deleting or
mismatching that link would be a data-integrity bug the database itself now
prevents, rather than something that could silently happen in a flat CSV.

### 8.3 Where the actual image files and the road graph live

Two kinds of data are deliberately **not** stored inside the database at all:

- **Uploaded photos** are saved as plain files under `data/uploads/`, named
  `{sha256_hash}_{original_filename}`. The database only stores the *path* to
  that file (`observations.image_path`), not the image bytes themselves.
  Putting large binary blobs directly into SQLite would bloat the database file
  and make every query slower to read off disk; storing a path instead is the
  standard approach, and keeps the database itself small and fast to query.
  The SHA-256 content hash doubles as a **deduplication check**
  (`file_hash_exists()` in `upload_pipeline.py`) — re-uploading the exact same
  photo is detected before it's ever saved twice.
- **The Sofia road network graph** is cached as a `.graphml` file
  (`outputs/geo/sofia_drive_network.graphml`), completely separate from the
  app database. This is static reference data pulled from OpenStreetMap, not
  something the app ever adds rows to — it's loaded once into memory
  (`get_sofia_graph()`) and reused for every matching/routing operation in a
  session, re-downloaded only if the cache file is missing or a refresh is
  explicitly forced.

### 8.4 Design principle running through all of it

The same idea shows up in both storage tracks and is worth being able to say
explicitly: **store the minimum that lets everything downstream be recomputed
correctly, and never store two conflicting copies of a derived number.**
`risk_score`/`risk_contribution` for a single detection is computed once and
stored; `road_segments.aggregated_risk` is *derived* from those detections and
is intentionally recomputed (not incrementally patched) every time a new
observation affects that segment, so it can never silently drift out of sync
with the detections it's supposed to summarize.

---

## 9. Quick reference — all the tunable numbers and where they live

| Parameter | Value | Meaning | File |
|---|---|---|---|
| `base_severity` | 1.0 / 1.2 / 2.0 / 3.0 | Fixed hazard weight per damage class | `config/risk_weights.yaml` |
| `confidence_floor` | 0.7 | Minimum confidence multiplier (at the threshold) | `config/risk_weights.yaml` |
| `confidence_threshold` | 0.25 | Confidence below which a detection isn't scored at all | `config/risk_weights.yaml` (must match app's detection threshold) |
| `beta` | 0.3 | Damping factor for extra (non-max) detections when combining | `config/risk_weights.yaml` |
| `observation_risk_max` | 3.0 | Ceiling used to normalize an image's score to [0, 1] | `config/risk_weights.yaml` |
| `risk_categories.low_max` / `medium_max` | 0.34 / 0.67 | Thresholds for low/medium/high | `config/risk_weights.yaml` |
| `routing_alpha` | 15000 | Distance-vs-risk trade-off in route cost | `config/risk_weights.yaml` |
| `max_matching_distance_m` | 50 | Max distance before a point needs manual road-matching review | `config/app_config.yaml` |
| `default_confidence_threshold` | 0.25 | Default YOLO inference confidence cutoff | `config/app_config.yaml` |
| Pilot area | Sofia, 8km radius | Scope of the geospatial/routing demo | `config/app_config.yaml` |

Every one of these can be explained to a mentor as: *"this is a simple, explainable
starting point, chosen either from domain reasoning (severity, confidence handling)
or empirically against the one real pilot dataset we have (alpha) — and it's kept
in one config file specifically so it can be revisited once real evaluation data is
available, without touching any code."*

---

## 10. The one-sentence version, if asked to summarize

*"Each detected piece of road damage gets a severity score based on what kind of
damage it is; multiple detections on the same photo or the same road combine using
a formula where the worst single hazard dominates but other damage still counts a
little; that combined score is what a routing algorithm uses as an extra 'cost' for
driving down a damaged road, so it can suggest a route that trades a bit of extra
distance for less exposure to known road damage — and every weight in that whole
chain lives in one config file, chosen for being explainable now and easy to
recalibrate later."*
