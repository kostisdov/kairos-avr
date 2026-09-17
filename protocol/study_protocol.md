# Study protocol

**KAIROS: dynamic prediction of structural deterioration of bioprosthetic aortic valves with competing mortality.**
Prototype development on synthetic scenarios, with a protocol for retrospective development and
independent validation in hospital cohorts.

Status tags used throughout: **Implemented** (exists in this repository and is tested),
**Demonstrated** (shown on explicitly synthetic scenarios), **Planned** (future work requiring
clinical cohorts), **Unverified** (stated by a source we could not check in full).

---

## 1. Title and objectives

### Primary objective

To develop, and in the next phase externally validate, a dynamic prediction model estimating the
1-, 3- and 5-year cumulative incidence of adjudicated structural valve deterioration (SVD) before
death, conditional on survival and valve status at prediction time, and to quantify the incremental
value of serial echocardiography, biomarker modules and anticoagulant-exposure history over
patient, valve and reference-echo information.

### Secondary objectives

Predict bioprosthetic valve failure, all-cause and SVD-attributable, as a separate outcome.
Characterise gradient and regurgitation trajectories by route and design class. Estimate the
association of antithrombotic exposure history with SVD and with thrombosis-related dysfunction as
distinct outcomes, reported as predictive associations. Test transportability to device generations
absent from training where event counts permit.

### Exploratory

Simulated counts of patients whose next assessment would be brought forward, under stated
assumptions. Longer-horizon estimation to 8 and 10 years by device generation. The framework for
15- to 20-year estimation in Appendix D of the specification. A nested imaging and pathology
component to attribute mechanism where CT or explant data exist.

### Out of scope

Any change to guideline surveillance intervals, any treatment recommendation, and any claim of
clinical accuracy before validation in real cohorts.

---

## 2. Target population

### Time zero

The date of the first adequate transthoracic echocardiogram performed 30 to 180 days after
implantation, the qualifying reference study. Patients enter the cohort at that date if they are
alive with the index valve and have not met the primary endpoint. Subsequent predictions are made at
each later echocardiogram and, for reminder purposes only, at implant anniversaries.

A fixed day-90 landmark was rejected: it cannot use an echo performed on day 150, and anchoring on a
date the data may not contain invites look-ahead leakage. Anchoring on the actual reference study
removes both problems.

### Inclusion

Adults with a first bioprosthetic aortic valve, surgical or transcatheter, implanted at a
participating centre, with a qualifying reference study. No criterion depends on what is observed
after entry.

### Exclusion at entry

Mechanical valves; endocarditis before the reference study; adjudicated deterioration or index-valve
replacement before the reference study; concomitant mitral prosthesis. Early procedural failures
before the reference study are described separately. Valve-in-valve implants form a pre-specified
sensitivity cohort.

### Prototype data and dates — Implemented

The de-identified records supplied for the event carry year-only dates. They cannot support landmark
timing, exposure lags or longitudinal rates. All demonstrations requiring day-level precision use
explicitly synthetic dates; the real records contribute extraction findings and descriptive counts
only.

### Cohort size for the planned study — Planned

With valve class and generation categories, spline terms, route-specific baselines and landmark
summaries, the core model carries roughly 40 to 60 effective parameters. Riley's criteria with an
anticipated Cox-Snell R² of 0.10 imply about 3,400 to 5,100 development patients, with the five-year
event fraction taken as 6% and 11% in two planning scenarios. Repeated landmark rows add no
independent patients and no independent events, so the sample size is counted in patients and in
adjudicated events, never in rows. External validation needs about 200 primary events for a precise
calibration slope. Numbers are recomputed with `pmsampsize` once endpoint prevalence is measured in
the source cohort.

---

## 3. Endpoints

| Outcome | Role |
|---|---|
| First adjudicated moderate or severe haemodynamic valve deterioration (VARC-3 stage 2 or 3) attributable to structural valve deterioration, stenotic, regurgitant or mixed | **Primary** |
| Bioprosthetic valve failure per VARC-3: stage 1, dysfunction with clinically expressive criteria or irreversible stage 3 haemodynamic deterioration; stage 2, aortic valve reintervention; stage 3, valve-related death. Reported as all-cause BVF and, where the underlying dysfunction is adjudicated as SVD, SVD-attributable BVF | **Key secondary** |
| SVD-related reintervention; time to first stage 3 haemodynamic deterioration | Secondary |
| Thrombosis-related dysfunction, endocarditis, non-structural dysfunction (patient-prosthesis mismatch, paravalvular leak, pannus, malposition) | Separately classified dysfunction. Thrombosis-related dysfunction is also modelled as its own outcome |
| Death from any cause before the primary endpoint | **Competing event** |
| Index-valve replacement or explant for a **non-SVD** cause before the primary endpoint | **Competing event**, because SVD of the index valve can no longer be observed |
| Reintervention performed for **suspected valve dysfunction** before deterioration has been adjudicated | **Counts as a primary endpoint event**, consistent with the VARC-3 bioprosthetic valve failure construct. Treating it as censoring would be informative censoring, because the valve is gone precisely because it was failing |
| Valve thrombosis treated medically and resolved | Intercurrent event: the patient remains at risk and the episode enters as a time-varying covariate |

### Clinical usefulness threshold — Planned

Calibration within 3 percentage points of observed at five years; Brier score better than the
reference model at 1, 3 and 5 years; discrimination above the reference as supporting evidence; and
positive net benefit across the 1-year risk thresholds at which an earlier assessment would be
triggered. Discrimination alone does not establish usefulness.

---

## 4. Data sources and ground truth

Implant record and device identifier. Echo reports, structured and free-text, for gradients,
velocities, area, dimensionless index, regurgitation, LVEF and stroke volume. Laboratory results by
module. Medications with start, stop, interruption and switch dates. Procedure and diagnosis codes
and death records for candidate outcomes. CT or explant findings where they exist. Access through
the institutional data warehouse under ethics approval, with text extraction for free text.

### Adjudication framework

Candidates are identified from codes and from echo reports flagged by the VARC-3 module; **codes
never establish a diagnosis.** Two reviewers independently classify each candidate using all
available evidence, with a third arbitrating.

Change-based staging requires the patient's own reference study; published normal values inform
context but cannot substitute for that baseline. **Patient-prosthesis mismatch present at the
reference study is not deterioration**: it is a fixed characteristic of the implant, and only change
from the patient's own baseline can constitute the endpoint.

A stenotic endpoint requires the gradient criterion together with the area or dimensionless-index
criterion. A regurgitant endpoint requires the regurgitation criterion and is not invalidated by a
missing area or index. Confirmation is a second study or, where that is impossible, operative,
pathological or unequivocal morphological evidence; death or reintervention before a second study is
a pre-specified exception.

Mechanism is assigned as SVD, thrombosis, endocarditis or non-structural with a confidence grade.
Where mechanisms overlap, the dominant mechanism is recorded with the alternative. Cases that cannot
be resolved are classed **uncertain**, never negative. Uncertain cases are excluded from the primary
analysis and included in a sensitivity analysis under both classifications.

---

## 5. Statistical analysis plan

### Primary model

Penalised cause-specific proportional hazards for SVD, for death and for non-SVD replacement, fitted
on a landmark dataset with one row per patient per prediction time, spline terms for continuous
predictors, baseline hazards stratified by route, and combined into cumulative incidence.

Device effects: design class as fixed effects, with model and generation where at least a
pre-specified number of events exist. A frailty term for model nested in class is the **planned**
extension and is described as planned wherever the hierarchy is mentioned.

### Comparator ladder

The reference model in the original specification was valve age and valve type alone. That is a weak
bar, and the comparison that decides clinical usefulness is against what a cardiologist already
does. The ladder is therefore:

| ID | Comparator | Information at the landmark | Role |
|---|---|---|---|
| R0 | Null | none | Anchor for the scaled Brier score and the reference line in decision curves |
| R1 | Valve age and valve type | route, design class | Secondary |
| R2 | Pre-implant | R1 plus patient factors and prosthesis characteristics, no echo | Secondary; also isolates the value of the reference echo |
| R3 | Plus the reference echo, fitted once, never refreshed | R2 plus the 30-to-180-day study | Secondary; isolates the value of updating |
| **R4** | **Dynamic single marker**: current mean gradient and its change from the reference study, refreshed at each landmark | | **Primary, statistical** |
| **R5** | **The guideline rule**: VARC-3 stage criteria applied to the current echo, given a calibrated risk by a one-covariate landmark model | | **Primary, clinical utility, decision curve** |
| R6 | A published external score | — | None exists; reporting standards require this to be stated |

KAIROS against R4 carries the incremental-value claim. KAIROS against R5 carries the net-benefit
claim, and R5 is the comparator that answers what the model adds to current practice. A matched
alert-rate table is reported: if the rule flags a given percentage of landmarks, the same top
percentage by model risk is taken and sensitivity and positive predictive value compared.

### Leakage rules — Implemented

Only information dated on or before the prediction time. Patients who met the endpoint leave the
risk set. **The echo that establishes an endpoint is never a predictor of that event.** All rows of
one patient stay together in every resampling. Preprocessing, imputation and model selection are
fitted within training data. The model chosen on development data is frozen before it meets the
temporal or external cohort, and no model is selected on the final cohort and then called validated.

### Population restriction for the primary analysis

Landmarks at which the patient already shows prevalent stage 1 or worse haemodynamic deterioration
are excluded from the primary analysis and reported separately. At those landmarks, crossing to
stage 2 is close to deterministic and the comparison against a threshold rule becomes circular. The
clinically useful population is the one in which deterioration has not yet declared itself.

### Horizon

Three years is the primary horizon. One year is reported alongside because it is the horizon the
supported decision needs, and five years for completeness. A one-year horizon alone would make the
single-marker comparator quasi-definitional, and that is stated rather than hidden.

### Observation process

The primary analysis predicts first adjudicated detection, which can be dated. Sensitivity analyses
treat onset as interval-censored between the last negative and first positive study, and model
surveillance intensity as informative. Missed visits are a reminder trigger and a confidence penalty;
their coefficient, if any, is estimated rather than imposed.

A **surveillance-blinded** variant is pre-specified: the identical model with time-since-last-echo
and the overdue indicator removed, reported beside the primary. If the advantage over R4 survives
their removal, the biological claim holds. If it does not, the model is a surveillance-triage tool,
which is still useful and must be described that way.

### Metrics, in priority order

1. Net benefit by decision curve, against R5, R4, treat-all and treat-none, at the three-year horizon.
2. Calibration for cumulative incidence: calibration-in-the-large, slope, and the integrated calibration index. Moderate calibration is demanded, not slope alone.
3. Brier score and its decomposition, inverse-probability-of-censoring weighted under competing risks, with paired bootstrap intervals.
4. Time-dependent AUC, cause-specific, reported per landmark and never as a single global number.
5. A single likelihood-ratio test on nested models.

Net reclassification improvement is not reported. Where a concordance index is requested, a
competing-risk-adapted version is used and the text states that concordance is improper for risk at
a fixed time.

### Optional comparison — Planned, lower priority

A gradient-boosting survival model fitted to the same cause-specific structure, reported only if it
beats the primary model on calibration and Brier score. It is not part of the core demonstration and
no other model family is promised.

### Validation sequence — Planned

Develop and tune in designated centres with patient-level resampling. Freeze. Evaluate in a later
implant cohort and in an untouched external centre. Run leave-one-generation-out checks where events
permit. Subgroups: route, design class, manufacturer, age at implant, sex, valve size and mismatch,
renal function, diabetes, failure phenotype.

---

## 6. Ethics and privacy

Retrospective, pseudonymised data. In Greece: hospital ethics committee approval, GDPR Article
9(2)(j) research basis, a data protection impact assessment, data minimisation and on-premises
training. For a US partner: IRB review with waiver of consent, HIPAA de-identification or a limited
dataset under a data use agreement. No identifiable data leaves the institution.

**Fairness.** Calibration and discrimination reported by sex, age band, race or ethnicity where
recorded, and by manufacturer, with recalibration by subgroup if drift is found.

**Transparency.** Each output shows the three probabilities, the horizon, the main drivers, the
reliability statement and, separately, any reminder. The model never recommends reintervention or a
treatment change and never alters guideline surveillance on its own. Reporting follows TRIPOD+AI
with a model card; model cards for the implemented models are in `docs/model_cards/`.

**Data handling in this repository — Implemented.** Patient-level material never enters version
control, a build context or a log. `scripts/privacy_scan.py` enforces this in CI and before every
deployment. The supplied spreadsheets are excluded by `.gitignore`; only code, aggregate counts with
small-cell suppression, and figures without patient-level rows are committed.

---

## 7. The decision the model supports

One decision only: whether the next assessment should be brought forward.

A predicted 12-month cumulative incidence of SVD at or above a pre-specified threshold triggers
clinical review with echocardiography at 6 months rather than at the next scheduled visit, **in
addition to the guideline schedule and never in place of it.** The threshold band is pre-specified at
5 to 15 percent, and the operating point is selected by decision-curve net benefit before results
are seen.

Three messages are kept apart and are never merged into one number:

- A **current abnormality** that needs clinical assessment now. This is a finding, not a prediction.
- A **predicted near-term risk** high enough to bring the next assessment forward.
- An **overdue scheduled echo**. This is a reminder and needs no model at all.

A five-year risk never drives an appointment interval. Guideline surveillance differs by
jurisdiction and is preserved as written in each: ESC/EACTS guidance recommends yearly clinical
review with imaging for bioprostheses, while the 2020 ACC/AHA guideline considers echo at 5 and 10
years and annually thereafter reasonable for surgical bioprostheses, and annual echo reasonable after
TAVI. The exact wording of the 2025 ESC/EACTS follow-up table is **Unverified** by us and is flagged
as such in `docs/kairos_unverified.md`.
