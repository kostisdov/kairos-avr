# Model approach

Model choice and justification, feature engineering, validation, and clinical integration.
Status tags: **Implemented**, **Demonstrated** (on synthetic scenarios), **Planned**.

> Every model output in this repository is trained on synthetic scenarios. It is illustrative and
> unvalidated. Synthetic performance is evidence about the demonstration, not about patients.

---

## Pipeline

| Stage | What happens | Where |
|---|---|---|
| Ingest | Implant records, echo reports, labs, medications | `src/kairos/io/` |
| Extract | Valve passport: model, size, route, date, serial gradients, dimensionless index, area, regurgitation, flow | `src/kairos/extraction/`, `src/kairos/passport.py` |
| Adjudicate | VARC-3 stage against the patient's reference study, with an uncertain class | `src/kairos/varc3.py`, `src/kairos/adjudication/` |
| Predict | Cause-specific hazards for SVD, death and non-SVD replacement at each landmark | `src/kairos/modelling/` |
| Update | Serial echo, biomarker and anticoagulant modules, as available | `src/kairos/modelling/modules.py` |
| Report | Three probabilities, drivers, reliability statement, separated messages | `src/kairos/modelling/predictor.py`, `services/predict/` |

---

## 1. Formulation

**Dynamic prediction by landmarking.** The dataset holds one row per patient per prediction time,
carrying every feature as known at that time. Cause-specific hazards for SVD, death and non-SVD
replacement over the following five years are estimated and combined into cumulative incidence,
conditional on being alive with the index valve and free of adjudicated SVD at the landmark.

Thrombosis-related dysfunction is modelled as its own outcome in a secondary analysis. Its
predictors differ from those of structural deterioration — route, intra-annular balloon-expandable
design, valve-in-valve, absence of anticoagulation — and so does its management. A resolved episode
returns the patient to the SVD risk set with the episode recorded as a covariate.

### Why this family and not another

The data forces the choice. Follow-up is censored, death competes and is common in this population,
events are rare, device categories are numerous and sparse, and clinicians have to be able to read
the model. A fixed-horizon classifier would discard the time information and misstate absolute risk
once a third of the cohort has died. A survival model does not remove the information shortage that
rare events impose; it only uses the information correctly.

---

## 2. The primary model

Penalised (ridge) cause-specific proportional hazards with spline terms, route-stratified baseline
hazards and landmark time as a covariate. Interpretable, publishable as a nomogram, and its
competing-risk claim is explicit in its structure rather than implied.

Device effects in the prototype are design-class fixed effects, with model and generation where the
scenario supplies enough events. The frailty term that would pool sparse devices toward their class
is **Planned** and is labelled as such wherever the hierarchy appears.

**No monotonic constraint on gradient change.** Gradients are flow dependent and failure may present
as regurgitation rather than stenosis. A constrained model is a pre-specified sensitivity analysis,
and performance is reported separately by failure phenotype.

**Challenger — Demonstrated.** A gradient-boosting survival model fitted to the same cause-specific
structure, promoted only if it beats the primary model on calibration and Brier score. A Cox-objective
boosted score alone yields a relative hazard, not a competing-risk probability, so the challenger
models death as well. No other model family is promised.

The comparison has been run and the challenger **did not earn promotion**. On the gradual-stenotic
scenario, with out-of-fold predictions on identical rows for both families and 200 bootstrap
replicates resampled by patient:

| Ladder step | Decision | Mean Brier difference | 95% interval |
|---|---|---|---|
| Reference | retain Cox | +3.4e-06 | [-6.4e-05, +6.5e-05] |
| Core | retain Cox | +2.0e-04 | [-2.9e-04, +7.0e-04] |
| Core plus both modules | retain Cox | +5.6e-04 | [-4.0e-06, +1.1e-03] |

10,701 landmark rows from 2,390 patients. A positive difference means boosting scored worse. Every
interval spans zero, so the two families are indistinguishable here and the first promotion criterion
fails at every step. The remaining criteria pass, so boosting is not worse beyond tolerance either; it
simply adds nothing on this scenario.

The result is published rather than dropped. A challenger that fails its own pre-specified promotion
rule is evidence that the rule was applied. Full decision records are in `docs/comparison/`.

---

## 3. Features

### Static, at time zero

Age at implant, sex, body surface area and body mass index, route, design class, model and generation
where available, label size within class, indexed effective orifice area and mismatch grade, tissue
treatment, reference mean gradient, dimensionless index, area, regurgitation and paravalvular leak,
LVEF and stroke volume index, diabetes with duration, eGFR and dialysis status, smoking status,
atrial fibrillation and any other anticoagulation indication, antithrombotic class at time zero,
bicuspid native valve, implant year, and for transcatheter implants the residual regurgitation at
discharge.

Lipid-lowering treatment is recorded as **exposure**, not as protection.

### Longitudinal, at each landmark

**Echo.** Current mean gradient; change from the reference study; most recent change and the interval
over which it occurred; longer-term slope where three or more studies exist; current and changed
dimensionless index and area; regurgitation grade and its change; stroke volume index and LVEF, so
that flow-dependent gradients are read in context; time since the last echo; and the overdue
indicator.

**Laboratory trajectories.** eGFR slope and any dialysis start date; HbA1c trajectory; NT-proBNP
trend. These are core features rather than an optional module: renal function, diabetes and the
ventricle's response to load are among the few predictors with bioprosthesis-specific evidence, and
their movement over time carries more than any single value.

Only measurements dated on or before the landmark are used.

### Missing data

Multiple imputation fitted within training data, for covariates measured in most patients. Markers
essentially absent in a cohort are **not imputed into existence**: the module that needs them is
switched off and the core model reports without it. Echo values older than 18 months are flagged as
stale, never carried forward silently.

---

## 4. Modules

Modules are feature blocks added to a working core one at a time, each judged on incremental
calibration and Brier score. No module is turned into a composite score, no marker is assumed to
help, and the core model runs when any module is unavailable.

### Biomarker modules

| Module | Markers | Status |
|---|---|---|
| Renal and metabolic | eGFR trajectory, dialysis, HbA1c and diabetes duration, LDL and non-HDL cholesterol with lipid-lowering treatment as exposure | Established association; incremental value to be measured |
| Mineral metabolism | Phosphate, albumin-corrected and ionised calcium, PTH, alkaline phosphatase | Hypothetical |
| Lipid-related susceptibility | Lipoprotein(a) in original units with assay recorded; oxidised phospholipids, Lp-PLA2, PCSK9 where measured | Hypothetical, phenotype-specific |
| Cardiac response | NT-proBNP | Hypothetical for SVD; mainly informs competing mortality |
| Vitamin K status | Dephosphorylated-uncarboxylated matrix Gla protein | Hypothetical; the single most informative research assay for this design |
| Inflammatory and molecular | hs-CRP; fetuin-A, calcification propensity; anti-alpha-Gal and anti-Neu5Gc IgG | Hypothetical |

Where a marker is essentially absent from a cohort, the prototype does not impute it and claims no
evidence about it. It shows how the pipeline would carry the marker and how the evaluation would
judge it.

### Anticoagulant exposure module

Treatment history modelled as dated, class-specific exposure rather than a discharge checkbox. Its
purpose is prediction: to learn whether exposure history carries information about structural
deterioration and, separately, about thrombosis-related dysfunction. It makes no treatment
recommendation and asserts no causal effect.

Recorded: drug and class (vitamin K antagonist, factor Xa inhibitor, direct thrombin inhibitor, with
antiplatelet agents separate); indication; start, stop, interruption and switching dates; cumulative
prior exposure by class; INR control only where adequately dated INR results exist; an indicator for
treatment initiated or changed within a defined window after a suspicious echo; and prescription
versus confirmed adherence as distinct fields.

**Reverse causation is handled explicitly, not by a blanket lag.** Treatment started because
dysfunction was suspected would otherwise be read as its cause. A fixed lag does not remove the
confounding by indication behind it, and would erase recent thrombosis information that the
thrombosis outcome needs. The module therefore keeps the post-suspicion indicator as its own
variable and reports associations with and without those exposures.

Pre-specified interactions: antithrombotic class by route, and cumulative vitamin K antagonist
exposure by tissue type.

Where dephosphorylated-uncarboxylated matrix Gla protein is measured, it enters beside cumulative
vitamin K antagonist exposure as the marker of the mechanism the exposure is presumed to act
through, so the model can separate "on warfarin" from "vitamin K depleted". The hypothesis is
directional: if the drug signal is real it should run through the marker, so the drug coefficient is
expected to shrink when the marker is added, and that shrinkage is itself a reported result. The
marker is never imputed across a cohort; the substudy model takes the full-cohort linear predictor
as an offset and estimates only the marker terms.

---

## 5. Outputs

One inference returns, at 1, 3 and 5 years plus a 12-month horizon:

- **Probability of adjudicated SVD before death.** The principal output.
- **Probability of death before deterioration**, and **probability of remaining alive with the index valve and free of deterioration.** These three, together with the small probability of non-SVD index-valve replacement reported separately, sum to one.
- The top drivers.
- A **reliability statement** covering device evidence, data completeness and staleness, and, where the model is trained on synthetic scenarios, the label *illustrative, unvalidated*.

**One minus the SVD cumulative incidence is not the probability of being alive with a working
valve.** It includes patients who died first. No output in this repository uses it that way.

Three messages are kept apart on screen: a current abnormality needing assessment now, which is a
finding rather than a prediction; a predicted near-term risk high enough to bring the next assessment
forward; and an overdue scheduled echo, which is a reminder. A five-year risk never drives an
appointment interval.

---

## 6. Validation

As in the protocol: develop and tune with patient-level resampling, freeze, evaluate in a later
implant cohort and an untouched external centre, then transportability checks.

**In the prototype — Demonstrated.** Patient-grouped cross-validation on synthetic scenarios, a
temporal split, and a leave-one-class-out run to exercise the code paths. Every metric is printed
with bootstrap intervals and every result is labelled synthetic.

The comparator ladder, the metric priority order and the surveillance-blinded variant are specified
in `protocol/study_protocol.md` §5 and are not repeated here.

---

## 7. Clinical integration

KAIROS runs when an echo report is finalised, and at implant anniversaries for reminders. Results
land on the valve-clinic worklist; a nurse coordinator books assessments; the structural heart team
reviews flags.

It never triggers a reintervention referral, never changes antithrombotic treatment, and never
lengthens a surveillance interval. Whether it improves attendance or finds deterioration earlier is a
question for the prospective phase, after independent validation and a silent evaluation.

The model does not choose between redo surgery and valve-in-valve, and will not be extended to do
so. The less invasive option is selected for anatomically suitable patients, so the comparison is
confounded before treatment and is not identifiable from observational data at any sample size. A
named treatment output also crosses the regulatory line that a risk probability does not. What the
model can legitimately show alongside the risk is the device information a heart team needs —
internal diameter, leaflet mounting, coronary obstruction risk class — which is a lookup from the
device table, not a prediction.

---

## 8. Limitations

New and withdrawn devices with short records. Sparse or irregular echo and the observation bias it
creates. Site variability in echo reading. Flow-dependent gradients and low-flow states. Regurgitant
failure that gradients miss. Mismatch mistaken for deterioration when the reference study is
missing. Valve-in-valve physiology. Extraction errors, above all native versus prosthetic values.
Confounding by indication and by era in every treatment covariate. Abrupt events with no preceding
signal, which no model can anticipate.

The prototype's synthetic training data is the largest limitation and is stated first wherever
results appear.
