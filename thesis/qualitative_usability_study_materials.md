# Qualitative Usability Study — Materials

Ready-to-use materials for a small (n=5-8) guided walkthrough + survey,
complementing the quantitative H1-H4 hypothesis testing with a qualitative
read on trust, clarity, and perceived usefulness (per the Research Quality
Guidelines' allowance for qualitative methods -- interviews, content
analysis -- to complement quantitative approaches).

---

## 1. Participant intro / consent line

Read or send this before starting (keeps it lightweight -- no formal
ethics board needed for a small BSc usability pilot, but consent should
still be explicit and in writing/recorded somewhere):

> Thank you for helping test this prototype for my BUas graduation
> project -- a road damage detection and risk-aware routing tool. This
> will take about 10-15 minutes: you'll try a few tasks in the app, then
> answer a short survey. Your responses are anonymous and used only for
> academic evaluation in my thesis. There are no right or wrong answers --
> I'm interested in your honest reaction, including anything that confuses
> you.

---

## 2. Task walkthrough script (~10-15 minutes)

Give the participant the app URL and this task list. If you're running it
live (in person or on a call), take brief notes on anything they hesitate
on, misread, or ask about -- this is qualitative data in its own right,
separate from the survey.

**Task 1 -- Detect damage**
Open the **Upload & Detect** page. Upload the provided sample photo (or
one of your own road photos, if you have one) and look at the detection
result and the risk score shown.

**Task 2 -- Explore the map**
Open the **Damage Map** page. Find a road segment on the map (click a few
markers/lines) and read its risk information in the popup/tooltip.

**Task 3 -- Compare routes**
Open the **Safer Route Planner** page. Use the preset test route (or
enter two points yourself) and compare the shortest route against the
"safer" route shown.

**Task 4 -- Explain it back**
Without looking anything up: in your own words, what do you think the
risk score and the colour coding (green/orange/red) mean?

---

## 3. Survey questions (paste directly into Google Forms)

**Section 1 -- About this session**
- Participant ID / initials (for your own tracking only, not shared): *(short text)*
- Have you used a similar road-condition or navigation app before? *(Yes / No)*

**Section 2 -- Closed questions (5-point scale: Strongly disagree -- Strongly agree)**
1. The detection results were easy to understand.
2. The risk score / category (low, medium, high) was intuitive.
3. The map made it clear which roads were risky.
4. I would trust this system's risk assessment.
5. The route comparison was useful for deciding which way to go.

**Section 3 -- Open-ended questions (short paragraph)**
6. What was confusing or unclear while using the app?
7. Would you trust an automated system like this to influence your route
   choice? Why or why not?
8. What would make this more useful to you?
9. Any other comments?

---

## 4. Where this fits in the thesis

- Methodology: describe this design (Section~\ref{sec:method-qualitative}
  in the LaTeX template -- already scaffolded).
- Results: descriptive statistics for Section 2's closed questions
  (mean/median per item) plus a short thematic summary of Section 3's
  open-ended answers (3-5 recurring themes, one or two representative
  quotes per theme).
- Discussion: connect back to H2 and H4 specifically -- "trust in the
  risk assessment" and "map clarity" are the qualitative counterpart to
  those two quantitative results.
