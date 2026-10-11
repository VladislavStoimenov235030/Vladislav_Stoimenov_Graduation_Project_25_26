# Qualitative Usability Study — Results & Thematic Analysis

n = 10 participants (P01-P10), none with prior experience of a similar
road-condition/navigation application. Raw data:
`thesis/Road_Guard_AI_Usability_Study_Responses_Qualitative_Results.csv`.
Descriptive stats: `outputs/reports/qualitative_survey_stats.csv` and the
chart `thesis/figures/qualitative_survey_means.png`.

## Closed questions (1-5 Likert scale)

| Question | Mean | Median | Min | Max |
|---|---|---|---|---|
| Detection results easy to understand | 4.2 | 4.0 | 4 | 5 |
| Risk score/category intuitive | 4.6 | 5.0 | 4 | 5 |
| Map made risky roads clear | 4.6 | 5.0 | 4 | 5 |
| Trust the risk assessment | 3.8 | 4.0 | 3 | 5 |
| Route comparison useful | 4.4 | 4.0 | 4 | 5 |

Four of five means are 4.2 or higher. The risk-category and map-clarity
questions are tied highest (4.6). Trust is lowest (3.8, minimum 3), and it
is the only dimension where participants qualified their answers with
conditions (Theme 4).

## Thematic analysis of open-ended responses

Seven recurring themes, identified across all open-ended questions (not
just the questions nominally about that theme) and the comprehension-check
question. Counts are numbers of participants who raised the point.

### Theme 1 — Accurate mental model of the risk colour-coding (comprehension check passed)

All 10/10 participants described the green, orange or yellow, and red
severity mapping correctly in outline, before any explanation was given in
the survey. Three participants (P03, P04, P10) described the score as
combining several detections rather than a single value, which is close to
the actual aggregation rule (Equation 3 in the thesis: the largest
detection plus a damped share of the rest). P10 wrote "summed", which
approximates the rule but is not exact.

> "I think based on all the damage on the road segment the risk is
> combined and multiplied." — P03
>
> "The risk score shown displays the cumulative risk of the road,
> because when I compared red and green segments, I saw that the red
> has highest risk because of the higher amount of damages." — P04
>
> "Risk score I think is the summed risk based on all detected damages
> and their types." — P10

This is direct qualitative evidence that the risk-scoring rule is
perceptibly simple: users reverse-engineered its logic from the interface
alone.

### Theme 2 — Interface and control clarity issues

Seven of ten participants (P01, P02, P03, P04, P05, P09, P10) reported
something unclear in the interface, from button visibility and marker
selection to the meaning of the controls. Three of them (P03, P04, P09)
named the Safer Route Planner specifically.

> "On the safer route planner page I found an adjustable variable
> called alpha for risk-aversion strength, which is not clear what
> it's used for." — P03
>
> "On the safer route planner page the function of buttons there was
> unclear. For example it took me some time to find where the map
> for choosing pins directly on it is." — P04
>
> "I was unsure how changing the risk-aversion setting would affect the
> recommended route." — P09

This is a concrete, actionable usability finding distinct from anything
the quantitative hypotheses could surface: the routing mechanism itself
works (H3 supported), but its controls are not self-explanatory to a
first-time user.

### Theme 3 — Onboarding and explanation are needed

Four participants (P03, P05, P07, P09) asked for guidance on what the
controls and codes mean: the alpha setting, the damage class codes, and
the thresholds.

> "The application will be more useful if it has something like a guide
> with deep explanations what is the function for every thresholds that
> can be set." — P03
>
> "I did not immediately understand the damage class codes shown with the
> detections." — P07
>
> "A brief introduction when opening the application would make it easier
> to use." — P07

This overlaps with Theme 2 but is a distinct request: an explanation or
onboarding layer, rather than a change to the controls themselves.

### Theme 4 — Trust is conditional, not unconditional

Every participant said they would use or consider the system, but six of
ten attached a condition. Two conditioned trust on detection accuracy:
P03 linked it to the per-class detection confidence, and P08 noted that
incorrect detections would affect confidence in the recommended route.
Others conditioned trust on how recent the imagery is (P06, P10) or on
being able to inspect the supporting images (P07). P09 conditioned use on
the safer alternative adding only a small distance.

> "It depends on the thresholds set. As I saw the detection confidence
> for some images and classes is low, which I wouldn't trust... will be
> useful for the classes that have highest confidence." — P03
>
> "I would use the results as guidance, but incorrect detections could
> affect my confidence in the recommended route." — P08

The accuracy-related conditions (P03, P08) corroborate, from an
independent, non-technical perspective, the detection-recall limitation
identified quantitatively in Section 6.3 (Discussion, H2). A participant
who had no knowledge of the model's per-class recall figures arrived at
the same concern from using the app.

### Theme 5 — Data currency and repair status

Four participants (P01, P06, P08, P10) raised how up to date the
information is: whether a segment has been repaired, when the images were
collected, and whether an unmarked road had been checked.

> "I was unsure whether a road without a marker had been checked or
> whether there was no information available." — P06
>
> "I would use it to help choose a route, but I would want to know how
> recently the road images were collected." — P06
>
> "Reports of repaired damage should be checked before the map is
> updated." — P08

P01 and P08 asked for a way to mark repaired segments, and P01 also asked
how an already-scanned segment's repair could be indicated.

### Theme 6 — Participants envision a real-time, deployed, mobile tool

Four participants (P01, P02, P05, P10) independently proposed moving from
the current batch-upload prototype toward real-time, in-vehicle use.

> "Connected to the gps location and... easily notify when you are
> approaching a high risk road segment." — P02
>
> "Made on a mobile app like the typical routing maps. This way you can
> plan and avoid damages on real-time when driving." — P05
>
> "Add location-based warnings before approaching damaged sections." — P10
>
> "If it gets the input from car video recorder which always records
> and scan the road, and can run all the collected videos through the
> pipeline." — P01

Related suggestions covered the full data lifecycle: marker aging/removal
once a segment is repaired (P01), and routing detections to the
responsible municipal authority automatically (P01, P04).

### Theme 7 — Correctly read as a prototype, evaluated accordingly

Two participants (P03, P04) explicitly framed their feedback in prototype
terms rather than judging it against a finished product. Earlier analysis
of five participants reported three; on the current responses, only these
two use prototype wording.

> "For a prototype app it's also really good... showing that everything
> is working perfectly." — P03
>
> "I think the prototype is great for now, and hope to be made fully
> operating in the future." — P04

## Summary for the thesis

The qualitative results are positive overall (four of five closed-question
means are 4.2 or higher, and trust is 3.8). More importantly, they are
substantive: all ten participants described the colour mapping correctly,
three inferred the aggregation logic without being told it, and the
conditions users attached to trust (detection accuracy and data currency)
match the limitations identified quantitatively. Users also asked for
interface clarification, onboarding and data-lifecycle features, and a
mobile, real-time deployment, none of which the quantitative hypotheses
anticipated.
