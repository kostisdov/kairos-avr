# KAIROS model changes: detailed development design

Date: 17 September 2026
Status: development design. Phases A (WP-A1 to WP-A6), B (WP-B1 to WP-B4) and C (WP-C0 to WP-C3, full comparison not run: compute budget pending) implemented and tested 17 September 2026 with the defaults of section 10, amended by the owner on 17 September 2026 so that no evaluation is refused for synthetic data (plan guard advisory, promotion rule always applied with an evidence label, boosting tuning falls back instead of failing: deviation 24); phases D and E not started. Phase B added one choice beyond this text: a covariate-free route-stratified baseline for causes between the baseline and covariate-model gates (deviation 22). Before/after tables are in `data_build_report.md`.
Parent: [kairos_model_change_design.md](kairos_model_change_design.md) (CR-02 to CR-08; review finding 1 stays excluded)
Audience: whoever implements the change set. The parent document says *what* and *why*; this one says *where, how, in what order, and how we know it is done*.

Conventions used here

- Paths are relative to the repository root. `file.py:NN` refers to the code as it stands on 17 September 2026.
- "Assumed" marks an engineering default that must be labelled as such in config and reports, exactly as the parent does.
- Work packages are `WP-<phase><n>`. Each has: purpose, code changes with signatures, config, tests, exit criterion.
- The scientific contract is unchanged: primary endpoint, reference echo at 30 to 180 days, horizons 1/3/5 years, four mutually exclusive probabilities, three separate messages, synthetic-only results.

---

## 1. Code audit: what the parent's findings look like in the code

This table is the reason for most of the design decisions below. Every row was verified by reading the file.

| # | CR | Location | Present behaviour | Consequence |
|---|---|---|---|---|
| F1 | CR-06 | `evaluation/ladder.py:69` | `module_availability(lm, cfg)` runs on the complete landmark table, then is passed into every `fit_step` call (`:86`, `:115`), into `secondary_thrombosis` and into `evaluate_vitamin_k` | Held-out rows influence which modules exist in the training fold |
| F2 | CR-06 | `modelling/modules.py:44-61` | Availability is the row fraction of the single anchor marker per module; when the anchor passes, every feature in the block enters | Frequent visitors dominate the denominator; an eligible phosphate lets all-missing calcium, PTH and ALP enter |
| F3 | CR-06 | `modelling/features.py:47` | `med = ... if x.notna().any() else 0.0` | An all-missing marker is manufactured as a constant zero plus an all-ones missing indicator (later silently dropped as constant by `cause_specific.py:58`) |
| F4 | CR-06 | `modelling/landmark.py:84` | With no exposure records, `ac_class_current = "none"`, `ac_current_status = "never"` | "No medication records" is encoded as verified absence of anticoagulation; the anticoagulant module's availability is always 100 % |
| F5 | CR-05 | `modelling/cif.py:24` | `d[:, 0] = 0.0` | Any hazard mass at the first grid point is discarded |
| F6 | CR-05 | `modelling/cif.py:27-32` | Total increment capped at 0.999 and rescaled | Arbitrary cap; renormalises substantive errors away |
| F7 | CR-05 | `modelling/cif.py:22`, `cause_specific.py:98` | `nan_to_num` and `maximum.accumulate` on hazards | Non-finite or decreasing hazards are repaired silently instead of rejected |
| F8 | CR-05 | `cause_specific.py:96` | `predict_cumulative_hazard(df, times=grid)` on a fixed weekly grid | lifelines evaluates the Breslow baseline with `np.interp` (`lifelines.utils.interpolate_at_times`), i.e. linearly between event times; the boundary convention is undocumented |
| F9 | CR-04 | `cause_specific.py:74-76`, `:91-93` | Fewer than 5 event **rows** sets the cause model to `None`, which then returns zero cumulative hazard | "Too few events" is served as "zero future risk" |
| F10 | CR-04 | `cause_specific.py:72`, `modules.py:82`, `ladder.py:84`, `metrics.py:149` | All event counts are landmark rows, not patients | One patient with six landmarks counts as six events; `n_events` in `ladder.csv` is a row count |
| F11 | CR-04 | `cause_specific.py:49,65`, `predictor.py:216` | Unseen route maps to the majority stratum; missing route is filled with `"SAVR"` | A missing route is silently substituted |
| F12 | CR-04 | `ladder.py:84-85` | A fold with under 5 SVD rows is skipped with `continue`; its rows stay NaN and are dropped by `ok` | Methods can be compared on different surviving subsets with no record |
| F13 | CR-04 | `artifacts/ladder_summary.md` | `high_competing_mortality` has 3 SVD event rows at 5 years with n = 2500 (SVD hazard falls with age, cohort mean age 84, death multiplier 2.5) | AUC 0.00 and slope -0.04 are reported as if estimable |
| F14 | CR-07 | `metrics.py:30` | `np.clip(G, 0.02, 1.0)` | Tiny censoring survival is clipped and the result called supported |
| F15 | CR-07 | `metrics.py:41-42` | Left limit implemented as `G(t - 1e-9)` | Works, but undocumented and fragile with tied times |
| F16 | CR-07 | `metrics.py:84-98` | Slope is an IPCW linear regression on the probability scale | Deviation 7; not the logistic recalibration slope |
| F17 | CR-07 | `metrics.py:171`, `:179` | Failed resamples are swallowed; an interval is reported from as few as 10 valid replicates | No record of bootstrap validity |
| F18 | CR-07 | `ladder.py:80,89` | Out-of-fold array holds SVD only | Death, replacement and no-event probabilities are never evaluated |
| F19 | sec. 2 | `generators.py:42` vs `adjudication/framework.py:30` vs `varc3.py:53` | Generator uses a five-level scale (0 none, 1 trace, 2 mild, 3 moderate, 4 severe); VARC-3 and the landmark builder use four levels (0 none/trace, 1 mild, 2 moderate, 3 severe). `generators.py:300` passes the five-level `ref_ar` into `Echo(regurg_grade=...)` unchanged; `:385` uses `min(ar, 3)` | A generator "mild" (2) is staged as "moderate" at reference inside the generator but as "mild" (1) in the landmark builder and predictor |
| F20 | sec. 2 | `generators.py:384-388` vs `predictor.py:190` | Synthetic label = first single study at stage 2/3 outside a true thrombosis episode; live endpoint = confirmed by a second study | Deviation 9; training labels and live eligibility follow different rules |
| F21 | CR-02 | `generators.py:150` | One `rng` drives static covariates, latent times, visits, noise and labs (only dp-ucMGP has its own stream, `:168`) | Changing the visit mechanism changes every later draw, including other patients' disease |
| F22 | CR-02 | `generators.py:402` | `svd_onset_date` is the start of progression (`t_onset`), not the crossing of the haemodynamic threshold | Detection-versus-onset comparisons use the wrong onset in the gradual scenario |
| F23 | sec. 2 | `landmark.py:255`, `generators.py:398-400` | `phenotype_latent`, `lpa_high_latent`, `dp_ucmgp_latent_log_implant` live in the `patients` table and `phenotype_latent` is copied into every landmark row | Not in any feature block today, but only by convention; nothing enforces it |
| F24 | sec. 2 | `landmark.py:222-223`, `framework.py:52` | Tabular NaN becomes `EchoPoint(mean_gradient=nan)`; `adequate` tests `is not None` | A NaN gradient counts as an adequate reference echo |
| F25 | sec. 2 | `generators.py:434-436`, `io/config.py:84` | Cohort prefix = hash(config_hash : seed); `config_hash` is the hash of the whole simulation section; git sha is `nogit` (the working directory is not a git repository) | A `--n 600` quick run overwrites the full cohort of the same seed; `nogit` is not a version |
| F26 | CR-08 | `predictor.py:87-100`, `cli.py:188` | `ModelBundle` has a hard `cox` field; `train` writes straight to `models/latest/` | No family abstraction; a failed or rejected run can overwrite the served bundle |
| F27 | CR-08 | `.venv` | `scikit-survival` is not installed. `pip install --dry-run scikit-survival` on this machine (Python 3.11.6, Windows) resolves to **0.28.0** with `ecos 2.0.14`, `osqp 1.1.3`, `numexpr 2.14.2` and no conflict with `scikit-learn 1.9.1` / `numpy 2.4.6` | Windows side looks clean; the Linux image still has to be verified before the pin is written (WP-C0) |

---

## 2. Target architecture

### 2.1 Package layout after the change

```text
src/kairos/
  grades.py                      NEW   single regurgitation grade mapping (F19)
  adjudication/framework.py      CHG   pure adjudication core + AdjudicationRecord, used by generator and predictor
  simulation/generators.py       CHG   named random streams, truth table, shared adjudication, manifest v2
  simulation/scenarios.py        CHG   effective_config_hash, visit-mechanism override hook
  modelling/
    landmark.py                  CHG   label policies, explicit exclusions, NaN hygiene, exposure coverage
    modules.py                   CHG   fit_eligibility / EligibilityManifest (training-fold only)
    features.py                  CHG   mode="cox"|"tree", refuses all-missing numeric features
    cif.py                       CHG   hazard contract + piecewise-constant-hazard integration
    base.py                      NEW   SurvivalFamilyAdapter protocol, breslow_step, errors, bundle v2
    cause_specific.py            CHG   CoxAdapter: step baseline, explicit unsupported fits, no route substitution
    gradient_boosting.py         NEW   CauseSpecificGradientBoostingModel (route x cause fits)
    tuning.py                    NEW   nested patient-grouped tuning for boosting
    explain.py                   NEW   grouped local sensitivity explanations (both families)
    train.py                     CHG   family-aware fit_step / train_bundle
    predictor.py                 CHG   adapter-based prediction, structured reliability, structured errors
  evaluation/
    support.py                   NEW   unique-event accounting, fit/metric/bootstrap gates, coverage
    metrics.py                   CHG   censoring models, IPCW logistic calibration, decomposition, bootstrap v2
    ladder.py                    CHG   family-aware harness, four-state OOF store, checkpoints
    compare.py                   NEW   paired differences, promotion gate, candidate freeze
    sensitivity.py               NEW   oracle / interval / missing-onset / visit-process analyses
    plots.py                     CHG   calibration curves at 1/3/5 y, family comparison, support panel
  extraction/schema.py           CHG   Reliability v2, Prediction family fields, ErrorDetail
services/predict/app.py          CHG   multi-family loading, ?family=, structured errors
services/demo/app.py             CHG   family selector, comparison view, reliability reasons
services/jobs/cli.py             CHG   --family, --families, --sensitivity, size-pilot, freeze-candidate, evaluate-final, publish
config/model.yaml                CHG   support, endpoint, eligibility, families, tuning, promotion, explanation groups
config/evaluation_plan.yaml      NEW   frozen sizes, seeds and tolerances (written once by the sizing pilot)
```

### 2.2 Data flow

```text
scenarios.yaml --> generate_cohort --> Cohort{patients, echoes, labs, exposures, events | truth}  + manifest v2
                                          |                                         \
                                          | (predictor-visible tables only)          \ (evaluation-only, never joined into features)
                                          v                                           v
                       build_landmark(label_policy) --> LandmarkBuild{rows, exclusions}     sensitivity.py
                                          |
                  per outer fold:         v
     fit_eligibility(train) -> FeaturePipeline(mode).fit(train) -> Adapter.fit(train)      [boosting: tuning.py on 3 inner folds]
                                          |
                                          v
        Adapter.cumulative_hazards(test, knot grid) -> cif.combine (PCH) -> 4-state OOF store (parquet, checkpointed)
                                          |
                                          v
      support gates -> metrics (IPCW Brier, logistic calibration, AUC, decomposition) -> paired bootstrap -> compare/promotion
```

Two families share everything except the box labelled `Adapter` and the `mode` of the feature pipeline.

---

## 3. Phase A: dataset, labels, availability

### WP-A1 Dataset identity and manifest v2

Purpose: parent section 2, first paragraph; fixes F25.

Changes

- `io/config.py`: add `source_tree_hash() -> str` (sha256 over the sorted relative path and bytes of `src/kairos/**/*.py`, `services/**/*.py` and `config/*.yaml`, first 12 hex chars) and `code_revision() -> dict` returning `{"kind": "git", "value": sha}` when a real sha exists, else `{"kind": "source_tree", "value": hash}`. `Settings.model_version()` uses it instead of `nogit`.
- `simulation/scenarios.py`: add `ScenarioSpec.effective_config_hash` = sha256 of the resolved `params` (today `config_hash` is identical for all scenarios because it hashes the whole section).
- `simulation/generators.py`:
  - `GENERATOR_VERSION = "2.0"`, `ENDPOINT_VERSION = "2"` (both change cohort content in this change set).
  - `build_manifest` adds: `schema_version: 2`, `endpoint_version`, `generator_version`, `effective_config` (resolved params), `effective_config_hash`, `code_revision`, `reference_hashes` (file sha256 of the device table and `prosthetic_valve_reference_values.csv`), `table_hashes` (sha256 of `pd.util.hash_pandas_object(df_with_iso_dates, index=False)` per table; parquet bytes are not stable enough to hash), `n_requested`, `n_retained`, `exclusions` (`died_or_replaced_before_reference`, ...), `namespace`.
  - `cohort_prefix(spec, seed, n, namespace)` = `{name}/{variant}/{namespace}/{tag}` with `tag = sha256(effective_config_hash : seed : n : GENERATOR_VERSION : ENDPOINT_VERSION)[:10]`. `namespace` is `quick` or `full`.
- `services/jobs/cli.py`: `latest.json` moves to `{name}/{variant}/{namespace}/latest.json`. `scenarios --n` without `--namespace quick` is refused when `n` differs from the frozen plan (WP-B2).

Tests (`tests/test_simulation.py`): two cohorts with the same seed and different `n` get different prefixes; manifest carries a non-`nogit` revision; identical inputs reproduce identical `table_hashes`.

Exit: a quick run cannot overwrite a full cohort; every artefact can be traced to code, config, seed and size.

### WP-A2 One regurgitation grade mapping

Purpose: parent section 2, last sentence of paragraph 3; fixes F19.

- New `kairos/grades.py`: `REGURG_ORDINAL = {"none": 0, "trace": 0, "trivial": 0, "mild": 1, "moderate": 2, "severe": 3}`, `regurg_ordinal(label) -> int | None`, `regurg_label(ordinal) -> str`. `framework.AR_ORDINAL` becomes a re-export.
- Generator: keep drawing *labels* (`none/trace/mild` at reference with the existing probabilities; `severe` after abrupt onset) and derive the ordinal only through `regurg_ordinal`. Delete `AR_LABELS`, `_ar_ordinal_to_label` and the `min(ar, 3)` patch.
- Test: for every label, generator staging, `landmark.echo_features` and `Predictor` see the same ordinal; a reference "mild" followed by "moderate" is a stage 2 regurgitant candidate in all three places.

### WP-A3 One adjudication policy for labels and live eligibility

Purpose: parent section 2, paragraph 3; closes deviation 9 (F20).

Design

- `adjudication/framework.py` gains a pure core that needs no pydantic objects:

```python
@dataclass(frozen=True)
class AdjudicationPolicy:
    version: str = "2"
    confirmation_required: bool = True
    terminal_event_confirms: bool = True     # death / reintervention before a second study is possible

@dataclass
class AdjudicationRecord:
    candidate_date: date | None              # first qualifying study of the surviving candidate
    confirmation_date: date | None           # second study, or the confirming terminal event
    adjudicated_date: date | None            # = candidate_date once confirmed (the proposal's detection date)
    mechanism: str                           # svd | thrombosis | endocarditis | uncertain
    confidence: str                          # high | moderate | low
    status: str                              # confirmed | confirmed_by_terminal_event | unconfirmed_pending
                                             # | uncertain_transient | thrombosis_attributed | none
    uncertain_dates: list[date]
    thrombosis_attributed_dates: list[date]

def adjudicate_series(points, ref, thrombosis_windows=(), mechanism_events=None,
                      death_date=None, reintervention_date=None,
                      policy=AdjudicationPolicy()) -> AdjudicationRecord
```

  The existing `adjudicate(...)` becomes a thin wrapper that converts `PassportEvent`s and maps the record onto `EndpointDecision`, so the extract service and its tests are untouched.
- Generator: delete the inline single-study detection (`:384-388`). After the visit loop, build `EchoPoint`s from the *observed* echoes, build thrombosis windows from the *synthetic exposure records* with the same rule the live path uses (`thrombosis_windows_from_exposures`, given a small date-typed adapter), and call `adjudicate_series`. The generator may no longer consult the true `active_thromb` flag for labelling; untreated true thrombosis now produces candidates that resolve and become `uncertain_transient`, as it would in real data.
- `events` table columns: `svd_candidate_date`, `svd_confirmation_date`, `svd_adjudicated_date`, `svd_mechanism`, `svd_confidence`, `svd_status`, `svd_uncertain_dates` (JSON). The old single-study date is kept as `svd_first_positive_date` for the legacy comparison only.
- Label policies, consumed by the landmark builder (WP-A5):

| Policy | Event | Patients with only an uncertain/unconfirmed candidate at date `u` |
|---|---|---|
| `primary` | `svd_adjudicated_date` when status is `confirmed*` | follow-up censored at `u`; no landmark on or after `u` |
| `sens_uncertain_positive` | as primary, plus `u` as an SVD event | event at `u` |
| `sens_uncertain_negative` | as primary | candidate ignored; patient stays at risk |
| `legacy_single_study` | `svd_first_positive_date` | n/a (version-labelled, for the before/after table only) |

- Live path: unchanged 409 rule (confirmed endpoint or replaced valve). An unconfirmed candidate still returns a prediction with `current_abnormality = true`. A passport with a prior `uncertain_transient` finding gets the patient-level reliability reason `prior_uncertain_finding` (the primary model was never trained on post-uncertain rows).

Tests (`tests/test_adjudication.py`, `tests/test_simulation.py`): generator and predictor given the same echo series return the same record; a single noisy positive followed by a normal study is `uncertain_transient` and not a training event under `primary`; a candidate followed by death without a second study is `confirmed_by_terminal_event`; the establishing echo is still never a landmark (existing leakage test stays green).

Exit: deviation 9 can be rewritten as "historical".

### WP-A4 Truth separation and named random streams

Purpose: parent section 2 paragraph 2 and section 7; fixes F21, F22, F23.

- `Cohort` gains a sixth table, `truth` (one row per patient, plus `truth_visits` long table), saved under `{prefix}/truth/`. It holds everything evaluation-only:
  `phenotype_latent`, `initiation_date` (today's `svd_onset_date`), `threshold_crossing_date` (new), `lpa_high_latent`, `dp_ucmgp_latent_log_implant`, `last_negative_date`, `first_positive_date`, `onset_interval_lo/hi`, `missed_crossing` (crossed before death/replacement/end but never detected); `truth_visits`: `scheduled_date`, `attended`, `eligible` (alive with index valve at that date), `noise_free_gradient/eoa/dvi/ar`.
  These columns are **removed** from `patients` and `events`. `build_landmark*` never receives `truth`.
- Noise-free threshold crossing: evaluate `stage_hvd` on the noise-free SVD trajectory (thrombosis contribution excluded) against the patient's reference echo on a weekly grid from `t_onset` to `t_end`, then bisect to the day. Regurgitant/abrupt: crossing = `t_onset`.
- Random streams: `np.random.default_rng([seed, STREAM_ID, patient_index])` per patient and stream, following the existing dp-ucMGP pattern (`generators.py:168`). Streams: `static`, `latent` (onset, death, replacement, progression slope, thrombosis episodes), `measurement` (echo noise, drawn for **every scheduled visit**, attended or not), `visits` (jitter and attendance), `labs`, `treatment` (post-suspicion decisions), `mgp`. Consequence: two cohorts that differ only in `surveillance.*` have identical `truth` latent columns and identical observed values at commonly attended visits.
- Attendance reporting: `five_year_attendance = attended / eligible` over scheduled visits in year 5 (deaths and replacements are not in the denominator).
- Guard: `modelling/landmark.py` exports `FORBIDDEN_PREDICTORS` (truth column names, `patient_id`, `landmark_date`, outcome columns, `bootstrap_cluster_id`); `fit_step` raises if a resolved feature is in it, and `build_landmark` raises if its inputs contain a truth column.

Tests: the paired-cohort identity above; `threshold_crossing_date >= initiation_date`; forced visit loss (attendance 0 for years 2 to 4) delays `svd_adjudicated_date` without changing any truth latent column; `FORBIDDEN_PREDICTORS` guard fires when `phenotype_latent` is injected into a block.

### WP-A5 Landmark builder: label policy, exclusions, NaN hygiene, unique events

Purpose: parent section 2 paragraphs 2, 4, 5; fixes F10, F24.

```python
@dataclass
class LandmarkBuild:
    rows: pd.DataFrame
    exclusions: dict[str, int]      # no_echo, no_reference, endpoint_at_or_before_reference,
                                    # death_or_replacement_before_reference, followup_end_before_reference
    label_policy: str
    endpoint_version: str

def build_landmark(cohort_tables, cfg, label_policy="primary", event_override: pd.DataFrame | None = None) -> LandmarkBuild
```

- `build_landmark_dataset(...)` stays as a wrapper returning `.rows` so existing callers and tests keep working.
- `event_override` lets `sensitivity.py` supply re-allocated event dates; eligibility is rebuilt from scratch for each allocation (rows on or after the allocated date disappear).
- Landmark exclusion is explicit for **each** of: endpoint date, death, replacement, end of follow-up (today death and replacement are excluded only because they equal `end_followup_date`).
- NaN hygiene: `_none_if_nan` on every `EchoPoint` field at construction; test that a NaN gradient is not an adequate reference.
- Exposure coverage (F4): `exposure_features` returns `ac_records_available: bool`; when false, the `ac_*` categorical features are `NaN` (one-hot level `missing`), cumulative years are `NaN`, not zero. Live semantics: `PredictRequest.exposure is None` means unknown coverage; an `ExposureTimeline` with an empty episode list means verified none. Document this in `docs/schemas/predict_request.json`.
- New columns: `bootstrap_cluster_id` (equals `patient_id` until a bootstrap duplicates the patient), `row_weight_patient_balanced = 1 / n_rows(patient)`.
- The "pooled over landmark rows" label is added to every metric table; the patient-balanced variant uses the weight above multiplied into the IPCW weights.

`evaluation/support.py` (first part, needed here):

```python
def unique_event_patients(lm, cause_code, by: str | None = None) -> int | dict   # a patient counts once if any row has that code
def event_rows(lm, cause_code, by=None) -> int | dict
```

`modules.event_count_rule` switches to `unique_event_patients` keyed on the original `patient_id` (never `bootstrap_cluster_id`).

Tests: three patients with four landmarks each yield 3 unique events and up to 12 event rows; bootstrap duplication does not raise the unique count.

### WP-A6 CR-06: training-fold eligibility

Purpose: parent section 3; fixes F1, F2, F3.

Config (`config/model.yaml`, replaces `module_availability`):

```yaml
eligibility:
  min_patient_fraction: 0.2            # label: assumed
  anchors:        {biomarker_renal_metabolic: [egfr], biomarker_mineral: [phosphate], biomarker_lipid: [lpa],
                   biomarker_cardiac: [ntprobnp], biomarker_inflammatory: [hscrp], anticoagulant: [ac_records_available]}
  markers:        {biomarker_renal_metabolic: [egfr, hba1c, ldl], biomarker_mineral: [phosphate, calcium_corrected, pth, alp],
                   biomarker_lipid: [lpa], biomarker_cardiac: [ntprobnp], biomarker_inflammatory: [hscrp]}
  derived:        {egfr_slope: {source: egfr, requires: dated_history}}   # eligibility judged on the derived column itself
  # block features not listed under markers/derived (dialysis_current, diabetes_duration, lipid_lowering, ac_*) follow the module status only
```

API (`modelling/modules.py`):

```python
@dataclass(frozen=True)
class MarkerEligibility:
    marker: str; n_patients: int; n_patients_measured: int
    patient_fraction: float; row_fraction: float; n_distinct: int
    eligible: bool; reason: str          # "", "below_cutoff", "absent", "constant"

@dataclass
class EligibilityManifest:
    version: str; cutoff: float
    markers: dict[str, MarkerEligibility]
    modules: dict[str, dict]             # status: full | partial | unavailable, included, excluded, reasons
    def card(self) -> dict

def fit_eligibility(lm_train: pd.DataFrame, cfg: dict) -> EligibilityManifest
def resolve_features(cfg, blocks, manifest) -> tuple[list[str], dict[str, str]]   # features, excluded feature -> reason
```

Rules: denominator = patients in `lm_train`; numerator = patients with at least one non-missing value at any of their landmarks. Anchor ineligible gives `unavailable` (whole block off). Anchor eligible: each remaining marker judged on its own; any exclusion gives `partial`. A measured-but-constant marker is removed with reason `constant` (distinct from `absent`).

`FeaturePipeline.fit` raises `ValueError` for an all-missing numeric feature; the `else 0.0` branch is deleted. `fit_step` loses its `availability` parameter and always fits eligibility on the rows it is given; the same goes for `secondary_thrombosis`, `evaluate_vitamin_k` and every inner tuning fit.

Inference: the bundle stores the manifest. New measurements of an excluded marker are ignored and listed under `reliability.excluded_features`; completeness is computed over trained eligible fields only.

Tests (`tests/test_eligibility.py`, new): the parent's three acceptance items, written as (1) eligible eGFR + all-missing HbA1c: HbA1c absent from `pipeline.output_columns_`; (2) fit on folds {0..3}, then mutate fold 4 arbitrarily: manifest, medians, levels and `rare_levels` byte-identical; (3) bundle round trip reproduces manifest and probabilities. Plus: ladder reports still list all 11 steps, with `equivalent_to` filled when two steps resolve to the same feature set.

**Phase A exit gate**: all existing tests green (149 collected on 17 September 2026) (with intended updates), new tests green, cohorts regenerated in the `quick` namespace, a before/after table of event counts under `legacy_single_study` vs `primary` written to `docs/data_build_report.md`.

---

## 4. Phase B: probability and evaluation

### WP-B1 CR-05: hazard contract and cumulative incidence

Purpose: parent section 4; fixes F5 to F8.

Contract (module docstring of `modelling/cif.py`, and the model card):

1. An adapter returns, per cause, a cumulative **integrated** hazard `H_k` of shape `(n, T)` on a strictly increasing grid with `grid[0] == 0` and `H_k[:, 0] == 0`.
2. Baselines are right-continuous step functions: `H0(t)` is the value at the last training event time `<= t`. No linear interpolation between event times.
3. The grid is `{0} ∪ horizons ∪ {near-term horizon} ∪ all baseline event times of every route/cause model <= max_cif_time_years`. Every jump therefore sits at the right end of exactly one interval, so no first-interval mass can be lost and horizons are exact grid points.
4. Combination uses the parent's piecewise-constant-hazard update. It is exact when hazards are constant within an interval; the survival convention is `S = exp(-sum_k H_k)` (Breslow), **not** the product-limit `prod(1 - dH)`.

```python
INTEGRATION_VERSION = "pch-expm1/1"

class HazardContractError(ValueError): ...

def validate_hazards(cum_hazards: dict, grid: np.ndarray, tol: float = 1e-9) -> dict
    # rejects: non-finite values, shape mismatch, non-increasing grid, grid[0] != 0, H[:,0] > tol,
    #          any decrease < -tol; clips decreases in (-tol, 0] and returns {"max_roundoff_correction": ...}

def combine_cause_specific(cum_hazards: dict, grid: np.ndarray, *, tol: float = 1e-9) -> dict
    # dH_k = diff(H_k); D = sum_k dH_k; S_prev = exp(-cumsum(D) shifted by one); q = -expm1(-D)
    # dF_k = S_prev * q * where(D > 0, dH_k / D, 0); F_k = cumsum(dF_k); alive_intact = exp(-cumsum(D))

def combine_product_limit_legacy(cum_hazards, grid) -> dict      # today's code, kept only for the comparison report
def combine_from_hazard_functions(h_funcs, grid, refine: int = 1) -> dict   # continuous hazards; used by tests and the reference check
```

Closure is algebraic (`S_prev * q = S_prev - S_next` and the shares sum to one), so `S + sum F = 1` holds to about 1e-15; the `1e-10` acceptance tolerance is comfortable and the predictor's "exact closure" patch (`predictor.py:232`) is replaced by an assertion.

Cox adapter mapping: `H_i(t) = step(cph.baseline_cumulative_hazard_[stratum])(t) * cph.predict_partial_hazard(x_i)`. lifelines centres covariates in both objects, so they are mutually consistent; a test pins this by comparing against `cph.predict_cumulative_hazard` **at event times** (where interpolation plays no role) to 1e-10.

Time support: each adapter reports `support_max_time[route][cause]` = the largest training time with at least `support.min_at_risk` (assumed 10) rows still at risk. A horizon beyond it yields the reliability reason `outside_training_followup`; the CIF is still returned but flagged, never silently flat.

A note for the implementer on the parent's "fine-grid numerical reference". For fitted step baselines the PCH result does not depend on refinement once all knots are in the grid, so a refined-grid check is vacuous there. The meaningful checks are:

- (a) **discretisation**: analytic continuous hazards (Weibull SVD, Gompertz death, exponential replacement, i.e. the generator's own forms) sampled on the grid, versus a numerical reference (`scipy.integrate.solve_ivp`, rtol 1e-10). Gate: max absolute horizon discrepancy < 1e-4; if it fails, `refine` is raised until it passes and the chosen value is stored in the bundle.
- (b) **convention**: legacy product-limit versus PCH on fitted hazards. This difference is of order `max dH` and is *reported* per scenario (max and mean at each horizon), not gated at 1e-4. It quantifies how much old metrics move because of the convention alone.

Tests (`tests/test_cif_metrics.py`, extended): every acceptance item of the parent, one test each: zero hazards; single cause `F = 1 - exp(-H)`; constant competing hazards closed form; large hazards (`H = 50`) without NaN; zero first interval; mass at the first knot is retained; tied event times across causes; exact horizon boundary; monotone `F_k`, non-increasing `S`; closure 1e-10; contract violations raise `HazardContractError`.

Exit: `INTEGRATION_VERSION` stored in every bundle and every metrics file; old metrics directories are never merged with new ones (the ladder loader refuses mixed versions).

### WP-B2 CR-04: support gates, sizing pilot, frozen plan

Purpose: parent section 6; fixes F9 to F13.

Config:

```yaml
support:                                # all label: assumed
  full:  {fit_min_unique_events: 30, metric_min_event_patients: 20, metric_min_control_patients: 20,
          slope_min_events: 50, bootstrap_min_valid: 100, bootstrap_min_success_fraction: 0.8,
          min_censoring_survival: 0.05, min_at_risk: 10}
  quick: {fit_min_unique_events: 5, metric_min_event_patients: 5, metric_min_control_patients: 5,
          slope_min_events: 10, bootstrap_min_valid: 20, bootstrap_min_success_fraction: 0.8,
          min_censoring_survival: 0.05, min_at_risk: 5}
```

API (`evaluation/support.py`):

```python
class SupportStatus(StrEnum): OK, INSUFFICIENT_EVENTS, INSUFFICIENT_FOLLOWUP, CENSORING_SUPPORT_FAILURE, FIT_FAILED

@dataclass
class SupportDecision:
    status: SupportStatus; gate: str; counts: dict; reason: str
    mode: str                     # full | quick ; quick decisions carry exploratory=True and can never promote
    exploratory: bool

def fit_gate(lm_train, route: str | None, cause: str, cfg, mode) -> SupportDecision
def metric_gate(time, event, patient_ids, cause_code, tau, metric: str, G_at_tau, cfg, mode) -> SupportDecision
def bootstrap_gate(n_requested, n_valid, cfg, mode) -> SupportDecision
def coverage(oof_status: pd.Series, lm: pd.DataFrame) -> dict      # share of intended rows/patients with valid OOF predictions
def common_evaluable_mask(*oof_frames) -> np.ndarray               # families are only ever compared on this intersection
```

Behavioural changes

- Adapters never return zeros for an unfitted cause. `fit` records a `SupportDecision` per route/cause; `cumulative_hazards` raises `UnsupportedFitError(route, cause, decision)` when asked for an unsupported one.
- Evaluation: a fold whose training set fails a fit gate gets OOF status `insufficient_events` for its rows (probabilities NaN). The ladder row still exists, with counts and reason. Metric gates apply independently per metric; a Brier with weak support is kept but labelled `exploratory` and excluded from winner selection.
- Release bundle: `train` refuses to publish a bundle unless every route/cause it claims has `OK`. The bundle's `claimed_routes` lists only supported routes; a request for another route is an applicability error (WP-D1), never a substitution (F11).
- Sizing consequence to expect: at n = 2500 the gradual scenario has 49 replacement event *rows* in total, so neither family can pass a 30-unique-event gate for replacement per route, and Cox only barely pooled. `high_competing_mortality` has 3 SVD event rows and a 90 % TAVR mix, so its SAVR SVD model cannot realistically pass at any affordable size. The pilot decides the sizes; the honest outcome for that scenario may be "TAVR route only; SAVR unmet gate reported".

Sizing pilot and frozen plan

- `cli.py size-pilot --scenario ... --n 2000 --dev-seeds 3`: generates development-only cohorts (seed family `9100+`, never reused), counts unique events per cause and route and per 4/5 training fold, and extrapolates the `n` needed for every gate with a 1.25 safety factor. It also times generation and one Cox and one boosting fit (input to WP-C0).
- Output `config/evaluation_plan.yaml`, written once and then hand-reviewed:

```yaml
frozen: true
plan_hash: <sha256 of everything below>
development: {seed: 20260916, n: {gradual_stenotic: ..., high_competing_mortality: ..., ...}}
final_test:  {seeds: [20261001, 20261002], n: {...}}          # generated only after freeze-candidate
hazard_parameters_hash: <effective_config_hash per scenario>
promotion_tolerances: {mean_brier_ci_excludes_zero: true, max_horizon_brier_worsening: 0.005, max_abs_cal_diff_worsening: 0.01}
```

- `evaluate` (full namespace) refuses to run when the plan is not frozen or a cohort's manifest does not match the plan. Any extra mortality multiplier is a new named variant in `scenarios.yaml` with label `assumed`, never an edit of the existing one.

Tests (`tests/test_support.py`, new): the parent's four acceptance items; plus `UnsupportedFitError` instead of zeros; quick-mode decisions are marked exploratory; `common_evaluable_mask` equals the intersection.

### WP-B3 CR-07: calibration and uncertainty

Purpose: parent section 5; fixes F14 to F18; closes deviation 7.

Censoring models (`evaluation/metrics.py`):

```python
class CensoringModel(Protocol):
    name: str
    def fit(self, time, event, X=None) -> Self
    def G_left(self, t, X=None) -> np.ndarray          # P(C >= t): evaluated with side="left" on the KM step function

class KMCensoring      # unconditional (primary); replaces censoring_survival(); no clipping
class CoxCensoring     # sensitivity; covariates [route, age_at_implant, t_lm, valve_age_years], fitted where evaluated

def ipcw_weights(time, event, tau, G: CensoringModel, X=None, sample_weight=None) -> WeightResult
    # WeightResult: w, ess = (sum w)^2 / sum w^2, max, p99, n_zero, G_at_tau, support_ok (G_at_tau >= min_censoring_survival)
```

`event_free_at` (administrative censoring at the horizon counts as observed event-free) is kept unchanged, and its regression test stays.

Outcome definition per state at horizon `tau`: for a cause `k`, `Y = 1` if that cause occurred by `tau`, `Y = 0` for another cause by `tau` or observed through `tau`. For `alive_intact`, `Y = 1` if observed event-free through `tau`, `Y = 0` for any event by `tau`. Weights are the same in both cases.

Formal calibration:

```python
@dataclass
class CalibrationFit:
    intercept_joint: float; slope: float          # logit P(Y=1) = a + b * logit(p)
    intercept_offset: float                       # calibration-in-the-large on the log-odds scale (offset model)
    status: str; reason: str                      # ok | no_events | separation | insufficient_support | not_converged
    n_events: int; ess: float; eps: float

def ipcw_logistic_calibration(pred, time, event, cause, tau, G, sample_weight=None, eps=1e-6) -> CalibrationFit
```

Implementation: a 30-line weighted IRLS (two parameters, then one parameter with offset). Not `sklearn.LogisticRegression`, which penalises by default and is slow inside a bootstrap. Separation is declared when IRLS fails to converge in 50 iterations, the Hessian is singular, or `|b| > 15`; the estimate is then missing with a reason, never zero.

Renames in result tables: `cal_large` becomes `obs_minus_pred` (descriptive, probability scale); new `cal_intercept_offset`, `cal_intercept_joint`, `cal_slope_logit`; the old linear slope survives as `cal_slope_prob_legacy_v1`.

Curves: at 1, 3 and 5 years for the SVD cause, (i) the existing binned Aalen-Johansen display (labelled "display"), (ii) a formal IPCW-weighted logistic fit on a three-knot restricted cubic spline of `logit(p)` with a patient-bootstrap band (labelled "formal"). A support panel shows the prediction histogram and the weight ESS.

Brier decomposition: with IPCW weights normalised to one over rows with known status and K = 10 quantile bins, report the five-term generalised decomposition (reliability, resolution, uncertainty, within-bin variance, within-bin covariance; Stephenson 2008), which reconstructs the normalised-weight Brier exactly. Test tolerance 1e-10. The table also shows the main `/n` Brier and the ratio of the two, since they differ by `sum(w)/n`.

Bootstrap v2:

```python
@dataclass
class BootstrapResult:
    intervals: dict[str, tuple[float, float] | None]
    n_requested: int; n_valid: dict[str, int]; failures: dict[str, int]; gate: dict[str, SupportDecision]
    kind: str            # "fixed_oof" (conditional on fitted models) | "full_refit"

def bootstrap_by_patient(fn, patient_ids, n_boot, seed, keys) -> BootstrapResult
def paired_bootstrap(fn_a, fn_b, patient_ids, n_boot, seed, keys) -> BootstrapResult   # same resamples, same evaluable mask
```

Keys: `brier`, `auc`, `ipa`, `obs_minus_pred`, `cal_intercept_offset`, `cal_slope_logit`. Failed resamples are counted by exception type; an interval is `None` when the bootstrap gate fails. The censoring model is refitted inside each resample (as today, implicitly).

Tests (`tests/test_calibration.py`, new): simulation with constant competing hazards and known `F_1(tau)`; calibrated predictions recover intercept near 0 and slope near 1 (mean over 20 seeds within 0.1); predictions distorted by `expit(a + b * logit(p))` recover slope near `1/b`; competing events stay controls; a resample keeps complete patient histories; no events gives `status = "no_events"`; decomposition reconstruction; the administrative-horizon regression test still passes.

### WP-B4 CR-02: observation-process and onset sensitivity

Purpose: parent section 7. Depends on WP-A3, A4, A5.

`evaluation/sensitivity.py`:

```python
ALLOCATIONS = ("left", "midpoint", "right", "uniform")

def onset_intervals(events, echoes, truth=None) -> pd.DataFrame
    # (last adequate negative after reference, first adjudicated positive]; falls back to (reference, first positive]
    # uses adjudication evidence only; truth is used solely to report discordance

def allocate_events(events, intervals, how: str, rng=None) -> pd.DataFrame      # event_override for build_landmark
def oracle_events(events, truth) -> pd.DataFrame                               # threshold_crossing_date as the SVD date
def missing_onset_events(events, truth) -> pd.DataFrame                        # patients with missed_crossing reclassified, labelled assumption
def run_observation_sensitivity(cohort, cfg, plan, families, steps) -> SensitivityResult
def paired_visit_cohorts(spec, n, seed) -> tuple[Cohort, Cohort]               # regular vs informative, identical truth
def visit_intensity_weights(lm_train, echoes_train) -> WeightModel             # Andersen-Gill gap-time model, stabilised, truncated at p1/p99
```

Fixed choices

- Same patients, same frozen outer folds for all four analyses. Steps: `reference`, `core`, and the deployed step. Both families; boosting reuses the hyperparameters selected in the primary analysis (no re-tuning inside sensitivities; stated in the report).
- `uniform` uses 20 draws; results are summarised as mean and range, labelled "assumption-based interval imputation", never "interval-censored likelihood".
- No allocated date may exceed death, replacement or end of follow-up; the function asserts it. The missing-onset analysis is the single, labelled exception the parent allows.
- Visit-intensity weights are fitted inside training folds only and come with positivity diagnostics (weight range, ESS, share truncated).

Reported: detection delay (adjudicated minus threshold crossing), missed crossings, discordant observed positives (noise-driven positives with no true crossing), primary versus oracle incidence, change in Brier and calibration per allocation, regular versus informative comparison, patient-balanced versus pooled.

Tests (`tests/test_observation_sensitivity.py`, new): the parent's five acceptance items, one test each, on a 300-patient cohort.

**Phase B exit gate**: Cox-only ladder rerun in the `quick` namespace under the new labels, integration and metrics; the convention-difference report from WP-B1 (b) exists; deviation 7 rewritten. No family comparison is interpreted before this gate (parent section 10).

---

## 5. Phase C: model families

### WP-C0 Dependency verification and compute benchmark (do this first in phase C)

- Verify `scikit-survival==0.28.0` on the Linux base image used by `services/*/Dockerfile` (`pip install --dry-run` inside `az acr build` or any Linux Python 3.11). Only then add it to `pyproject.toml` (`scikit-survival>=0.28,<0.29`) and regenerate `requirements.lock.txt`; all four images install from the same lock, so all four get it. Record the verified versions in deviation 12.
- Benchmark one boosting fit (300 trees, depth 2) on a development fold of the pilot-sized cohort. The full comparison costs roughly:

  `fits = scenarios(9) x steps(11) x outer(5) x [grid x inner(3) + 1] x route-cause models(6)`

  With the parent's 8-point grid that is about 74 000 fits; with staged trees (below) about 38 000. Cox needs 9 x 11 x 5 x 3 = 1 485 fits and today takes about 165 s per scenario.
- Mitigations designed in: (1) **staged trees**: fit `n_estimators = 300` once per `(learning_rate, max_depth)` and score the 100-tree model from `staged_predict` plus our own Breslow baseline (`base.breslow_step`), halving the grid to 4 fits; (2) `joblib.Parallel` over `(step, outer fold)`; (3) per-fold checkpoints (WP-C3) so the job can resume; (4) the expensive run is a dedicated job, never part of `pytest`.
- If the benchmark still projects more than the agreed wall-clock budget, stop and ask the owner before shrinking the grid or the step list; both are frozen by the parent.

### WP-C1 Adapter protocol and bundle v2

Purpose: parent section 9 "Estimator and interface", last paragraph; fixes F11, F26.

`modelling/base.py`:

```python
CAUSES = {"svd": 1, "death": 2, "replacement": 3}

class UnsupportedFitError(Exception): ...      # carries route, cause, SupportDecision
class UnsupportedRouteError(Exception): ...    # route missing or not in bundle.claimed_routes
class LegacyBundleError(Exception): ...

class SurvivalFamilyAdapter(Protocol):
    family: str                     # "cox" | "gradient_boosting"
    adapter_version: str
    def fit(self, X: pd.DataFrame, time, event, route: pd.Series, patient_ids, *, support_cfg, mode) -> Self
    def knots(self) -> np.ndarray                                   # union of baseline event times
    def cumulative_hazards(self, X, route, times) -> dict[str, np.ndarray]   # hazard contract of WP-B1
    def log_risk(self, X, route, cause) -> np.ndarray               # used as the dp-ucMGP offset
    def support(self) -> dict                                       # (route, cause) -> SupportDecision, support_max_time
    def training_summary(self) -> dict

def breslow_step(time, event_indicator, log_risk) -> tuple[np.ndarray, np.ndarray]    # knots, cumulative baseline
def step_eval(knots, values, times) -> np.ndarray                                      # right-continuous
```

`ModelBundle` v2 (stays in `predictor.py`, joblib stays the format):

```python
schema_version: int = 2
family: str; adapter: SurvivalFamilyAdapter; pipeline: FeaturePipeline
eligibility: EligibilityManifest
semantic_features: list[str]; transformed_columns: list[str]
grid: np.ndarray; integration_version: str; refine: int
claimed_routes: list[str]; support: dict
hyperparameters: dict; tuning_record: dict | None
reference_profile: dict                 # for explanations (WP-D2)
bundle_evidence: dict                   # held-out synthetic evidence by phenotype, used for bundle-level reliability reasons
dataset: dict                           # manifest hashes of the training cohort, label_policy, endpoint_version
library_versions: dict; config_snapshot: dict   # full model.yaml + evaluation_plan hash
vitamin_k: DpUcMgpOffsetModel | None
```

`ModelBundle.load` raises `LegacyBundleError` for an object without `schema_version` (the `.cox` bundles). Recommendation: **reject, do not migrate**; the integration convention and labels both change, so old probabilities are not reproducible by design. `load` also refuses a bundle whose `library_versions` differ in major/minor from the runtime for `scikit-survival`, `scikit-learn` or `lifelines`.

`CoxAdapter` (`cause_specific.py`): today's estimator (pooled over routes, route-stratified baseline, ridge) behind the protocol; step baseline per WP-B1; `UnsupportedFitError` instead of `None`; no majority-stratum substitution. `contributions()` remains as the optional Cox-only view.

dp-ucMGP: `fit_vitamin_k` uses `adapter.log_risk(..., "svd")` as the offset for both families. For boosting the per-route additive constant is absorbed by the route-stratified offset model that already exists.

Tests: round trip with identical probabilities for both families; legacy bundle rejected with a clear message; unknown route raises `UnsupportedRouteError`.

### WP-C2 `CauseSpecificGradientBoostingModel`

Purpose: parent section 9.

```python
@dataclass
class CauseSpecificGradientBoostingModel:        # family = "gradient_boosting"
    n_estimators: int; learning_rate: float; max_depth: int
    min_samples_leaf: int = 20; subsample: float = 1.0; random_state: int = 20260917
    models_: dict[tuple[str, str], GradientBoostingSurvivalAnalysis]   # (route, cause)
    dropped_constant_: dict[tuple[str, str], list[str]]
```

- One `GradientBoostingSurvivalAnalysis(loss="coxph", n_iter_no_change=None)` per route and cause; `y = Surv.from_arrays(event == code, time)`, so other causes are censored at their time. Early stopping stays off because its internal split is by row and would separate a patient's landmarks.
- Input: `FeaturePipeline(mode="tree")`: same eligibility, same median imputation and missing indicators, same one-hot levels and rare-level rule, but **no** standardisation and **no** spline basis. `route` columns are dropped inside each per-route fit; other columns constant within a route are dropped and recorded. `t_lm` and `valve_age_years` are always kept. The bundle records both `semantic_features` and `transformed_columns`, which is how "identical information" is demonstrated.
- Hazards: `H_i(t) = step_eval(model.unique_times_, model.predict_cumulative_hazard_function(x_i))` at the shared knot grid; zero before the first event time of that model. A test pins `breslow_step(time, event, model.predict(X))` against the library output to 1e-8, which licenses the staged-tree shortcut in tuning.
- The model card states the extra flexibility: per-route risk functions and baselines, versus Cox's shared coefficients with stratified baselines. Thrombosis stays a secondary outcome, not a fourth cause.

Tests (`tests/test_gradient_boosting.py`, new; marked `slow`, run on a 400-patient cohort with 30 trees): fit/predict contract; hazards satisfy `validate_hazards`; four probabilities close; unsupported route/cause raises; determinism under a fixed seed.

### WP-C3 Nested tuning and the evaluation harness

Purpose: parent section 9 "Tuning and fair evaluation".

Config:

```yaml
families:
  cox: {penalizer: 0.05, l1_ratio: 0.0, tuned: false}
  gradient_boosting:
    grid: {n_estimators: [100, 300], learning_rate: [0.03, 0.1], max_depth: [1, 2]}   # label: assumed
    fixed: {min_samples_leaf: 20, subsample: 1.0}
    inner_splits: 3
    objective: mean_ipcw_brier_3causes_3horizons
evaluation: {n_splits: 5, bootstrap: 200, seed: 20260916, quick_bootstrap: 30, quick_grid: {n_estimators: [50], learning_rate: [0.1], max_depth: [1]}}
```

`modelling/tuning.py`:

```python
def tune_boosting(lm_outer_train, blocks, cfg, seed) -> TuningRecord
    # inner folds = patient_folds(lm_outer_train.patient_id, 3, seed + outer_k)
    # every inner fit refits eligibility, pipeline, rare levels and baselines on the inner-train rows only
    # score = mean over 3 causes x 3 horizons of IPCW Brier on inner-test rows
    # a tuple is invalid if any route/cause fit is unsupported or fails, or a probability contract check fails
    # winner = lowest mean score among valid tuples; ties within 1e-6 resolved by (max_depth, n_estimators, learning_rate) ascending
```

One shared tuple is chosen for the whole competing-risk system of that outer fold. The record keeps every tuple's per-cause, per-horizon scores, so the owner can see how much the unscaled mean is dominated by death (the most frequent cause); an IPA-scaled variant is *recorded* for information and never used for selection.

`evaluation/ladder.py` becomes family-aware:

```python
def evaluate_family_ladder(cohort, cfg, plan, family, steps=None, run_dir=None, n_jobs=1) -> LadderResult
```

- Fold assignment is computed once per cohort (`patient_folds`, unchanged) and written to `folds.parquet`; both families read it.
- OOF store, one parquet per `(scenario, family, step, fold)`: `row_id, patient_id, landmark_date, fold, horizon, p_svd, p_death, p_replacement, p_alive_intact, status, reason`. The file's presence plus a content hash of `(cohort table hashes, config hash, family, step, fold)` is the checkpoint; a rerun skips it.
- `ladder_v2.csv` columns: `scenario, family, step, equivalent_to, state (svd|death|replacement|alive_intact), horizon_years, weighting (pooled_rows|patient_balanced), n_rows, n_patients, n_event_patients, n_event_rows, n_control_patients, coverage_rows, coverage_patients, support_status, support_reason, observed, mean_predicted, obs_minus_pred(+CI), brier(+CI), brier_null, ipa(+CI), cal_intercept_offset(+CI), cal_slope_logit(+CI), cal_slope_prob_legacy_v1, auc(+CI), boot_requested, boot_valid, integration_version, endpoint_version, label_policy, label`.
- Timings, selected hyperparameters per outer fold, training-only decisions (eligibility manifests, rare levels) and failure reasons go to `run.json`.
- The phenotype breakdown joins `truth.phenotype_latent` at evaluation time (it is no longer in the landmark table) and passes through the metric gates instead of the ad hoc `mm.sum() < 30`.

`evaluation/compare.py`:

```python
def paired_differences(oof_cox, oof_gb, lm, cfg, plan) -> pd.DataFrame      # on common_evaluable_mask, paired patient bootstrap
def promotion_decision(diffs, plan) -> dict                                  # applies the frozen tolerances; AUC is never a criterion
def freeze_candidate(run_dir, decision) -> Path                              # candidate.json: family, step, hyperparameters, plan hash, run id
def evaluate_final(candidate, plan) -> dict                                  # generates the final-test cohorts, evaluates once
```

Promotion rule as frozen in the plan: paired mean SVD Brier difference across horizons with a 95 % interval excluding zero in boosting's favour; no supported horizon worse by more than 0.005; `|obs_minus_pred|` not worse by more than 0.01; logit slope estimable. Failure keeps Cox as default and still publishes the boosting report. `evaluate_final` refuses to run without `candidate.json`, and refuses a second run for the same candidate unless `--rerun-reason` is given (the reason is written into the report). A new seed from the same generator is reported as synthetic replication, not external validation.

Tests (`tests/test_nested_cv.py`, new): no patient appears on both sides of any inner or outer split; mutating outer-test rows leaves the tuning record unchanged; checkpoint resume yields identical OOF files; a rejected candidate leaves `active.json` untouched.

---

## 6. Phase D: outputs, serving, jobs

### WP-D1 CR-03: reliability, applicability and the error contract

`extraction/schema.py` (all new fields optional with defaults, so existing clients keep working):

```python
ReasonCode = Literal["abrupt_failure_not_reliably_anticipated", "phenotype_performance_unestablished",
                     "insufficient_event_support", "outside_training_followup", "unknown_device_or_route",
                     "stale_echo", "module_unavailable", "prior_uncertain_finding", "exposure_coverage_unknown"]

class ReliabilityReason(BaseModel):
    code: ReasonCode; severity: Literal["info", "warning", "blocking"]
    scope: Literal["bundle", "patient"]; message: str; detail: dict = {}

class ModuleStatus(BaseModel):
    status: Literal["full", "partial", "unavailable"]; included: list[str]; excluded: dict[str, str]

class Reliability(BaseModel):            # existing: device_evidence, data_completeness, stale_echo, label
    reasons: list[ReliabilityReason] = []
    applicable_scope: str | None = None  # e.g. "gradual stenotic deterioration; synthetic training only"
    model_family: str | None = None
    evaluation_support: dict | None = None
    modules: dict[str, ModuleStatus] = {}
    excluded_features: list[str] = []

class Prediction(BaseModel):             # adds
    model_family: str | None = None; integration_version: str | None = None; bundle_schema_version: int | None = None

class ErrorDetail(BaseModel):
    code: str; message: str; detail: dict = {}
```

Rules in `Predictor._reliability`

- Bundle-level reasons come from `bundle.bundle_evidence` and the training scenario: a bundle trained only on gradual stenosis carries `abrupt_failure_not_reliably_anticipated` and `phenotype_performance_unestablished` for **every** patient. Nothing is inferred from simulator truth or a scenario name at request time.
- Patient-level reasons use contemporaneous inputs only: `stale_echo`, `module_unavailable`, `exposure_coverage_unknown`, `prior_uncertain_finding`, `outside_training_followup`.
- A new regurgitant abnormality on the latest echo sets `current_abnormality` regardless of the predicted risk (already true through `stage_point`; pinned by a test).
- `data_completeness` is computed over trained eligible semantic features (F3/F4 no longer count manufactured values as observed).

HTTP contract (`services/predict/app.py`): `detail` becomes an `ErrorDetail` object everywhere.

| Status | `code` | When |
|---|---|---|
| 409 | `endpoint_met` | unchanged rule (confirmed endpoint or replaced valve) |
| 400 | `no_reference_echo`, `prediction_time_invalid` | unchanged conditions, now distinguishable by code |
| 422 | `unknown_device_or_route`, `unsupported_route`, `unsupported_fit` | route missing or not in `claimed_routes`; blocking support failure |
| 503 | `no_model_bundle` | unchanged |

The demo reads `detail.message` (falling back to a plain string). Tests in `tests/test_predict_service.py` are updated accordingly.

Invariance tests (`tests/test_reliability.py`, new): appending future echoes, flipping `truth.phenotype_latent` or changing future event labels leaves today's prediction and reliability byte-identical; warnings are present for both families; 409 still fires.

### WP-D2 Grouped local sensitivity explanations

`modelling/explain.py`:

```python
def grouped_sensitivity(predict_p12m: Callable[[pd.DataFrame], np.ndarray], row: pd.DataFrame,
                        groups: dict, reference_profile: dict) -> list[GroupEffect]
    # builds one frame with len(groups)+1 rows (original + one per replaced group), one transform, one hazard call
```

Config `explanation_groups` in `model.yaml`. Two reference kinds, declared per group:

- `own_reference_echo` for serial-echo groups, which keeps them internally consistent: gradient group `{current_gradient := ref_gradient, delta_gradient := 0, last_change_gradient := 0, gradient_slope := 0}`; likewise EOA, DVI and regurgitation groups. Reading: "what the 12-month SVD probability would be if this measurement had not changed since the reference study".
- `training_reference` for static, biomarker and exposure groups: training median (numeric) or mode (categorical), stored in `bundle.reference_profile`.

`Driver` gains optional `delta_probability: float` and `method: str`; `direction` is the sign of `p(actual) - p(reference)`. The text beside the table says these are model sensitivities, not treatment effects. The same method serves both families in the comparison view; Cox log-hazard contributions stay as a separately labelled optional panel.

Test: for a Cox bundle the sign of each group effect agrees with the sign of the summed coefficient contribution in a monotone single-feature group; the reported delta equals a manual recomputation.

### WP-D3 Predict service and demo

- Store layout (fixes F26):

```text
models/runs/{run_id}/{family}/bundle.joblib, card.json, demo_patient.json
models/{namespace}/{family}/active.json     {run_id, path, sha256, previous: {...}, published_at}
metrics/runs/{run_id}/...                   figures/runs/{run_id}/...
```

  `publish` writes `active.json` only after a run finished and its gates allow it; `previous` keeps the last pointer for rollback (`cli.py publish --rollback --family cox`). `models/latest/` is no longer written.
- Predict service: loads every family that has an `active.json` in its namespace. Default family = `KAIROS_MODEL_FAMILY` env, else `deployed_model.family` in `model.yaml` (initially `cox`). `POST /predict?family=gradient_boosting` and `/predict/trajectory?family=` are allowed only for loaded, compatible bundles (same endpoint version, horizons and integration version as the default); otherwise 422. `GET /models` lists families with version, step and support summary; `/model?family=`; `/healthz` lists loaded families. Family choice never alters eligibility rules, horizons or messages; they live in `Predictor`, not in the adapter.
- Demo (`services/demo/app.py`): sidebar family selector; a "Compare families" tab showing the four probabilities at 1/3/5 years side by side, the 12-month risk, the grouped sensitivities for both, and family plus version next to every number; the reliability panel lists reason codes with their messages and shows excluded modules separately from completeness. In line with the owner's standing expectation for the demo, every input stays editable and the comparison reruns on edit. The concise abrupt-failure sentence sits directly beside the numerical output, and the three messages remain three separate elements.

Tests: `tests/test_predict_service.py` (family parameter, 422 codes, `/models`), `tests/test_demo_ui.py` (selector present, reasons rendered, comparison tab).

### WP-D4 Jobs CLI and `build_all.py`

| Command | New behaviour |
|---|---|
| `scenarios` | `--namespace quick|full`, `--visit-mechanism`, plan check, truth tables saved |
| `size-pilot` | WP-B2 |
| `train` | `--family cox|gradient_boosting` (default: both), writes to `models/runs/{run_id}/`, prints support decisions, never publishes |
| `evaluate` | `--families cox gradient_boosting`, `--all`, `--sensitivity observation`, `--resume RUN_ID`, `--n-jobs`, `--quick` |
| `compare` | paired differences and promotion decision for a run |
| `freeze-candidate` | writes `candidate.json` |
| `evaluate-final` | one-shot final-test evaluation |
| `publish` | `--family`, `--run-id`, `--rollback` |

`build_all.py --quick`: small cohorts in the `quick` namespace, both families, `quick_grid`, relaxed gates, every output stamped `smoke test: not evidence`. The full path runs the frozen plan with checkpoints. Record peak memory and wall time in `run.json`; Azure job resources are changed only if those numbers require it.

---

## 7. Phase E: tests, documentation, release evidence

### 7.1 Test inventory

| File | Status | Covers |
|---|---|---|
| `tests/test_simulation.py` | extend | manifest v2, prefix identity, streams, truth separation, attendance denominator |
| `tests/test_adjudication.py` | extend | `adjudicate_series`, generator/predictor agreement, label policies |
| `tests/test_landmark_leakage.py` | extend | explicit exclusions, NaN hygiene, `FORBIDDEN_PREDICTORS`, exposure coverage |
| `tests/test_eligibility.py` | new | CR-06 acceptance |
| `tests/test_cif_metrics.py` | extend | CR-05 acceptance, contract errors, administrative-horizon regression |
| `tests/test_support.py` | new | CR-04 acceptance |
| `tests/test_calibration.py` | new | CR-07 acceptance, decomposition, bootstrap v2 |
| `tests/test_observation_sensitivity.py` | new | CR-02 acceptance |
| `tests/test_gradient_boosting.py` | new, `slow` | adapter contract, Breslow equivalence |
| `tests/test_nested_cv.py` | new, `slow` | grouping, training-only tuning, checkpoints, candidate safety |
| `tests/test_reliability.py` | new | CR-03 invariances, reason codes |
| `tests/test_model_pipeline.py`, `test_predict_service.py`, `test_demo_ui.py`, `test_vitamin_k.py`, `test_schema.py` | update | bundle v2, error contract, family fields, exported schemas |

`pytest -q` runs everything except `slow`; `pytest -m slow` runs in `build_all.py` and before any publish. Lint, `scripts/privacy_scan.py` and `scripts/export_schemas.py` stay in the build.

### 7.2 Documentation updates

- `docs/deviations.md`: rewrite entries 7, 9, 14 and 16 as "historical behaviour / completed change". Entry 14: the Lp(a) sub-distribution ratio is relabelled `assumed sensitivity parameter` in `scenarios.yaml`. Add entries for: (19) PCH/Breslow survival convention replacing product-limit, with the measured convention difference; (20) interval-imputation sensitivity instead of an interval-censored likelihood; (21) boosting's per-route risk functions versus Cox's shared coefficients; (22) unscaled mean Brier as tuning objective; (23) primary handling of uncertain candidates (censor at the candidate); (24) synthetic replication is not external validation.
- `docs/schemas/*.json` regenerated; `docs/runbook.md` gains the new command sequence, the rollback procedure and measured resources; a model card per family (scope, support, integration version, explanation method, limits).
- `docs/kairos_unverified.md`: Azure runtime acceptance and the hosted extraction comparison stay listed as unverified; this change set does not touch them.

### 7.3 Release evidence checklist (maps to parent section 11)

1. All tests, lint, privacy and schema checks clean.
2. Frozen `evaluation_plan.yaml` with its hash in every run.
3. Full report: both families x 11 steps x 9 scenario variants x 3 horizons x 4 states, with support status, coverage, unique patient and event counts, eligibility exclusions, bootstrap validity and fit failures shown even when unmet.
4. Paired differences and the promotion decision; `candidate.json`; one final-test report.
5. Sensitivity report (four analyses).
6. Old calibration conclusions replaced only by regenerated metrics; no file mixes integration or endpoint versions.

---

## 8. Sequence, dependencies, effort

```text
A1 -> A2 -> A3 -> A4 -> A5 -> A6 --+
                                   +--> B1 -> B2 -> B3 -> B4 --+
                          C0 (can start any time) -------------+--> C1 -> C2 -> C3 -> D1 -> D2 -> D3 -> D4 -> E
```

| Phase | Rough effort (focused days) | Compute |
|---|---|---|
| A | 4 | minutes (quick cohorts) |
| B | 5 | about 30 min for the Cox-only rerun at today's sizes; more at pilot sizes |
| C | 5 | dominated by the boosting comparison; hours to tens of hours depending on the WP-C0 benchmark |
| D | 3 | negligible |
| E | 2 plus the full run | full evaluation job with checkpoints |

Effort figures are estimates for planning, not commitments. Phases A and B must be complete before any family comparison is interpreted.

## 9. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Boosting comparison exceeds the compute budget | high | WP-C0 benchmark first; staged trees; parallel folds; checkpoints; owner decision before any reduction of the frozen grid or steps |
| Replacement and high-mortality SVD events cannot reach the gates at affordable n | high | sizing pilot; report unmet gates truthfully; claim only supported routes |
| Confirmation-based labels cut event counts noticeably (last-visit candidates, noisy single positives) | medium | before/after table in WP-A3; feeds the sizing pilot |
| `scikit-survival` wheel or transitive pins differ on the Linux image | low to medium | verify before pinning (WP-C0); one lock for all images |
| Regenerated cohorts change every published number | certain | version stamps on every artefact; loaders refuse mixed versions; change log entry |
| joblib bundles break across library upgrades | medium | `library_versions` check at load; retrain rather than migrate |

## 10. Decisions needed from the owner

Each has a default the implementation will follow unless told otherwise.

1. **Uncertain candidates in the primary analysis**: default = censor at the uncertain candidate date (positive and negative handling as sensitivities). Alternative = keep the patient at risk, as the live adjudicator does.
2. **Legacy `.cox` bundles**: default = reject with a retrain message. Alternative = write a migrator (not recommended; probabilities would differ anyway).
3. **Tuning objective**: default = unscaled mean IPCW Brier over 3 causes x 3 horizons, as the parent states, with the scaled version recorded for information.
4. **Sensitivity refits**: default = three steps (reference, core, deployed) and no re-tuning inside sensitivities.
5. **Compute budget** for the full comparison (wall-clock hours, local machine versus an Azure job), to be set after the WP-C0 benchmark.
6. **High-mortality scenario**: accept "TAVR route only, SAVR gate unmet" as a valid reported outcome, or add a separately named assumed variant with a more balanced route mix.
