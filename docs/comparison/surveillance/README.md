# KAIROS against current practice

Synthetic and illustrative, out-of-fold throughout. Produced by
`scripts/evaluate_surveillance.py --scenario <name>` (design in
`src/kairos/evaluation/surveillance.py`, settings in `config/clinical_comparison.yaml`). One folder per
scenario, each with a full `report.md`.

## Why the first attempt measured nothing

The earlier comparison (`../clinical_rule/`) scored the VARC-3 rule and the model on the same
landmarks and got no verdict from the rule on any of them. Two things were wrong:

1. The synthetic echoes did not say where regurgitation came from, so the rule could never rule
   out regurgitant deterioration and abstained. Generator 2.2 now reports the location on 90 percent
   of echoes (assumed) and records a paravalvular leak grade separately.
2. Even with complete inputs, the rule is negative at every landmark by construction. Landmarks stop
   before the first echo that meets the endpoint, so the moment the rule fires the event has already
   been dated. The rule is a detector and the model a predictor, so they have to be compared **over
   time**, as surveillance policies.

## The question

If KAIROS may bring the next echo forward when a patient's 12-month SVD risk is high, how much sooner
is deterioration caught, and how many extra echoes does that cost? Guideline echoes are never removed.

Each held-out patient's noise-free valve trajectory is replayed under two guideline schedules:

- **annual**: ESC/EACTS, a yearly echo for every bioprosthesis;
- **ACC/AHA**: surgical valves at 5 and 10 years then yearly, TAVI yearly.

Each schedule is also run with KAIROS guidance at thresholds of 2, 5 and 10 percent. Detection is
the first echo on which the VARC-3 rule is positive against the patient's reference study.

## Results (95% intervals from 200 patient-bootstrap replicates)

| Scenario | Policy | Mean lead, months (95% CI) | Deteriorations caught only with KAIROS | Extra echoes per 1,000 patient-years |
|---|---|---|---|---|
| Gradual stenotic | annual + KAIROS 2% | **1.8 (1.2 to 2.6)** | 12 (base only: 7) | 17 (+2%) |
| Gradual stenotic | ACC/AHA + KAIROS 2% | 0.1 | **20 (base only: 1)** | 8 |
| Regurgitant abrupt | annual + KAIROS 2% | 0.6 (0.3 to 1.0) | 6 (base only: 3) | 36 |
| Regurgitant abrupt | ACC/AHA + KAIROS 2% | 1.5 (0.6 to 2.7) | 14 (base only: 0) | 35 |
| Irregular surveillance | annual + KAIROS 2% | 0.7 (0.4 to 1.1) | 11 (base only: 6) | 14 |
| Irregular surveillance | ACC/AHA + KAIROS 2% | 0.1 | 15 (base only: 1) | 8 |

What this says, stated plainly:

- **Under a yearly schedule the gain is modest:** about two months earlier on average in the gradual
  scenario, and the slowest tenth of detections improves from 14.6 to 10.5 months after the valve
  actually crosses the threshold. Most patients are caught on the same echo either way.
- **Under the ACC/AHA calendar the gain is in who gets caught at all.** Surgical valves are imaged at
  5 and 10 years, so only 30 percent of deteriorating surgical valves are detected before death,
  replacement or the end of follow-up in the gradual scenario. KAIROS at 2 percent raises that to 43
  percent for about 11 extra echoes per 1,000 surgical patient-years.
- **The protocol's 5 to 15 percent band is too high for a 12-month risk.** Baseline 12-month SVD
  incidence is about 0.3 percent, so almost no patient reaches 5 percent; at 5 percent the policy
  barely differs from the calendar, and at 10 percent not at all. The useful thresholds sit at 1 to
  3 percent, where net benefit on the landmark decision curve is also positive. The alert threshold
  should be re-chosen for the 12-month horizon, or the band applied to a longer horizon.
- **The cost is more than echoes:** positives on echoes before any real crossing (noise or
  thrombosis) rise from 4.5 to 5.5 per 1,000 patient-years in the gradual scenario at 2 percent.

## As a risk score (12-month SVD, primary population)

| Scenario | Model AUC | Valve age and type AUC | Calendar AUC |
|---|---|---|---|
| Gradual stenotic | 0.94 | 0.71 | 0.61 |
| Regurgitant abrupt | 0.71 | 0.67 | 0.60 |
| Irregular surveillance | 0.85 | 0.71 | 0.49 |

Read the gradual-stenotic 0.94 with care. The synthetic label is a gradient threshold and the model
reads gradients, so part of that discrimination is built into the data. The primary population
excludes landmarks with a gradient already 10 mmHg above the reference study, or regurgitation up a
grade (83 of 10,701 in the gradual scenario). VARC-3 stage 1 is morphological and not simulated, so
this is a haemodynamic proxy. The model's 12-month calibration slope is above 1 (1.6 to 1.7) in two
scenarios, meaning its 12-month risks are too compressed.

## Not run

`high_competing_mortality` has too few SVD events for the model's support gates (three in the ladder
evaluation), so no comparison is possible and none is reported.

## Assumptions (all labelled assumed)

Full attendance in the replayed schedules. A single positive study counts as detection. The
regurgitation location is reported on 90 percent of echoes. An echo is brought forward to 6 months.
Guidance uses the `core` model, which has no anticoagulant block, because the cohort's prescriptions
were triggered by the original echo stream. The simulated patients follow the generator's own model of
deterioration, so this shows what KAIROS can do if that model is right. It is not clinical evidence.

The policy simulation follows the design of team CardioNTUA and the calendar comparator that of team
dyanooumenoi (Dyania Health Hackathon 2026).
