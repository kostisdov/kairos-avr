# KAIROS model changes: implementation design

Date: 17 September 2026  
Status: design for implementation; changes below are not yet implemented  
Request: exclude review finding 1; incorporate findings 2–7 and add gradient boosting.

## 1. Scope and decisions

This document supplements [the revised proposal](kairos_proposal_revised.html) and [the Azure build design](kairos_azure_build_design.md). It specifies corrections to synthetic data construction, probability calculations, evaluation and reliability, and makes a gradient boosting survival comparison a required deliverable. Penalised cause-specific Cox remains the primary reference implementation and the default served model until the comparison gates below are satisfied.

Review finding 1 (Storage network access) is excluded at the owner's request. This change set contains no Storage networking work. Exclusion is a scope decision, not a claim that network settings have changed.

| Change | Requirement | Completion evidence |
|---|---|---|
| CR-02 | Observation-process and interval-censoring sensitivity | Detection/onset/interval analyses, informative-visit diagnostics and reproducible scenario results |
| CR-03 | Abrupt/regurgitant applicability warning | Structured reliability reasons in API, model card and demo without use of future phenotype |
| CR-04 | Adequate high-mortality evaluation | Unique-event counts, explicit estimability gates, larger prespecified cohorts and supported metrics |
| CR-05 | Validated CIF calculation | Shared hazard convention, stable integration, analytical and numerical reference tests |
| CR-06 | Feature-level availability | Training-fold-only eligibility, absent-marker exclusion and reproducible module manifests |
| CR-07 | Formal calibration assessment | IPCW logistic recalibration, calibration intervals, curves and support diagnostics |
| CR-08 | Gradient boosting survival models | Three competing causes, route-specific fits, complete ladder, nested patient-grouped comparison and versioned bundles |

The primary endpoint, reference echo at 30–180 days, prediction horizons, four mutually exclusive outcome probabilities and three separate messages remain the governing scientific contract. Real year-only records remain extraction/descriptive data; they must not become training rows with invented dates. All reported model results remain synthetic, illustrative and unvalidated.

Gradient boosting was previously optional and lower priority. This request supersedes that priority only: implementation and comparative reporting are now required. A poor result must still be reported; it does not justify suppressing the comparison or automatically replacing Cox.

## 2. Shared training-data contract

Both model families consume the same frozen cohort tables and landmark rows. Extend the dataset manifest with schema version, endpoint version, generator version, effective configuration, seed, source/reference-table hashes, table hashes, requested and retained patient counts, exclusions and code revision. If Git is unavailable, store a source-tree hash instead of treating `nogit` as a unique version. Include cohort size in the artifact identity so a quick run cannot overwrite a full cohort generated with the same seed/configuration.

Keep `patient_id`, landmark date and outcome fields as audit metadata. Never include patient identifiers, latent onset/phenotype, future visit attendance, future medication decisions or endpoint confirmation information as predictors. Select the reference study and enforce exclusion at/before entry, at endpoint, death, replacement and follow-up end explicitly. Validate missing values as missing, including NaN in tabular echo inputs.

Resolve endpoint consistency before regenerating results. The current deviations log records single-study synthetic labels but confirmation-based live endpoint decisions. Store candidate date, confirmation date, adjudicated detection date, mechanism, confidence and uncertain status separately. Use a common adjudication policy for training labels and live eligibility, retaining the proposal's first qualifying detection date after confirmation. An unconfirmed candidate can still produce the separate current-abnormality message; it must not silently become a confirmed training label. Uncertain cases have an explicit primary exclusion and positive/negative sensitivity classification. Map regurgitation grades through one shared ordinal mapping in generation, staging and prediction.

Count unique patients with a cause-specific event for event-support rules, including device/generation eligibility. Repeated landmark rows do not create independent events. Report both event-bearing rows and unique event patients. Bootstrap duplicates retain their original patient identity for eligibility counts and receive a separate bootstrap cluster identifier when needed for fitting.

Preserve the existing prediction-at-echo estimand. Do not add anniversary training rows merely to make overdue/stale indicators vary; anniversary messages remain reminders. Label performance pooled over landmark rows as such, and add a patient-balanced sensitivity using equal total weight per patient to assess informative visit frequency.

## 3. CR-06: marker and module availability

Implement an eligibility stage fitted separately within every training fold, including inner folds used for tuning. The existing `evaluate_ladder` computes availability on the complete landmark table before splitting; remove that dependency on held-out observations.

For each biomarker, record eligible training patients, patients with at least one available measurement at an eligible landmark, measured row fraction, measured patient fraction and distinct measured values. Use patient-level coverage as the eligibility denominator so frequent visitors do not determine availability. Retain 20% as the initial configurable, explicitly assumed cutoff.

Drop a marker below the cutoff before imputation or feature transformation. Never manufacture an all-missing marker with a zero median. Retain median imputation and missing indicators for eligible markers with individual missing values, fitted only on training rows. Distinguish a measured constant marker from an absent marker and record its removal for lack of variation.

Module availability follows explicit configured required anchors; within an available module, assess every remaining marker independently. For example, an available phosphate anchor must not cause absent calcium, PTH or ALP to enter by imputation. Persist `full`, `partial` or `unavailable` status, included/excluded features, coverage and reasons. Derived marker features such as slopes require sufficient dated history, not just a nonmissing source marker. Apply the same principle to exposure history: no medication records means unknown coverage, not verified absence of anticoagulation.

At inference, reuse the bundle's frozen eligibility and preprocessing. New measurements do not silently activate untrained features. Missing required patient inputs reduce completeness and produce a reason; supported modules can continue using their trained missingness strategy. All ladder steps still appear in reports, with disabled modules and equivalent feature sets identified.

Acceptance: an eligible eGFR must not enable an all-missing companion marker; changing only the held-out fold must not alter training eligibility, medians, encodings or device-event thresholds; loaded bundles reproduce eligibility and predictions.

## 4. CR-05: cumulative incidence and numerical contract

The prior review's numerical concern requires precision. Product-limit integration is a legitimate estimator when increments represent discrete conditional event probabilities. The current implementation instead consumes sampled Cox cumulative hazards, clips total increments and tests primarily probability closure. Closure alone does not establish accuracy. Define one explicit convention for both model families and verify it against independent examples.

For this change set, use nonnegative cumulative integrated hazards with a piecewise-constant competing-hazard approximation. On each interval, let `dH_k` be the increment for cause `k`, `D = sum(dH_k)` and `S_prev` the probability of no event at interval start:

```text
q = -expm1(-D)
dF_k = S_prev * q * dH_k / D       when D > 0
dF_k = 0                         when D = 0
S_next = S_prev * exp(-D)
F_k_next = F_k_prev + dF_k
```

This update is exact for constant cause-specific hazards within an interval; it is not a claim of exact recovery of an arbitrary continuous hazard. Document how each adapter maps its fitted baseline hazard onto these intervals. Include zero, requested horizons and all relevant baseline event knots, and support optional interval refinement. Evaluate step-function baselines with a documented boundary convention rather than arbitrary linear interpolation. Do not discard nonzero hazard mass at the first event time.

Reject nonfinite hazards, materially decreasing cumulative hazards, mismatched shapes and invalid time grids. Allow only tiny round-off correction, recording tolerance. Remove the arbitrary `0.999` increment cap and do not renormalise away substantive errors. Flag horizons beyond reliable training follow-up instead of silently extending a flat tail as if it had evidence.

Acceptance tests: zero hazards; one cause with `F=1-exp(-H)`; constant competing hazards with `F_k(t)=lambda_k/sum(lambda)*(1-exp(-sum(lambda)*t))`; large hazards; zero first interval; tied event times; exact horizon boundaries; monotonicity; and `S+sum(F)=1` within `1e-10`. Compare the legacy product-limit calculation, the new convention and a fine-grid numerical reference on synthetic and fitted hazards. Target maximum absolute horizon discrepancy below `1e-4` against the reference; fail or refine if exceeded. Store integration method/version in every bundle and rebuild all old metrics after the change.

## 5. CR-07: calibration and uncertainty

At each cause and horizon `tau`, define `Y=1` for that cause by `tau`. Subjects experiencing another competing event before `tau` have `Y=0`, with known status thereafter. Subjects observed event-free through `tau` also have `Y=0`; earlier censoring leaves the status unknown.

Use IPCW weights `1/G(T-)` for observed events by `tau`, `1/G(tau-)` for subjects observed through the horizon, and zero for earlier censored subjects. Preserve the existing administrative-horizon censoring regression test. Record the censoring estimator, left-limit convention, weight distribution, effective sample size and support failures. The current unconditional KM estimator assumes independent censoring; assess that assumption and use a specified conditional censoring model in relevant sensitivity analyses. Do not silently clip tiny censoring survival values and call the result supported.

For supported predictions, estimate the logistic calibration model:

```text
logit(P(Y=1)) = alpha + beta * logit(predicted CIF at tau)
```

Fit the weighted binary likelihood using IPCW. Ideal joint intercept/slope are zero and one. Also fit an intercept-only model with the prediction logit as an offset for calibration-in-the-large on the log-odds scale. Retain the existing observed-minus-mean-predicted probability difference as a separately named descriptive metric; do not confuse it with the logistic intercept. Clip probabilities only for the logit transform with a declared epsilon. Separation, no events or insufficient support produce a missing estimate and reason, not a successful zero slope.

Use patient-cluster bootstrap confidence intervals for the probability difference, offset intercept, joint slope, Brier, AUC and paired model differences. Report failed/degenerate resamples and interval availability. Fixed out-of-fold prediction bootstrap measures evaluation uncertainty conditional on those fitted models; separately label any full-refit bootstrap that includes model fitting uncertainty. Add censoring-aware calibration curves at 1, 3 and 5 years and distinguish binning displays from formal calibration estimators. Preserve the old probability-scale slope only as a version-labelled legacy field.

The IPCW logistic approach is supported by the primary methods paper [Calibration plots for multistate risk predictions models](https://arxiv.org/abs/2308.13394). Competing-risk evaluation definitions should also be checked against [Validation of prediction models in the presence of competing risks](https://www.bmj.com/content/377/bmj-2021-069249).

Acceptance: simulations with known calibrated and deliberately distorted risks recover approximately ideal and nonideal intercept/slope respectively across repeated seeds; competing events remain controls; patient-cluster intervals use complete patient histories; both causes and no-event probabilities receive evaluation appropriate to their outcome definition. Add a documented Brier reliability/resolution/uncertainty decomposition with censoring-aware estimates and check its reconstruction tolerance; the existing null Brier and IPA are not a decomposition.

## 6. CR-04: rare-event support and high mortality

Preserve the existing high-mortality cohort as a named stress variant. Its reported three SVD events are insufficient evidence of comparative model quality. Prefer increasing the prespecified number of simulated patients before altering scientific assumptions. Any additional mortality multiplier becomes a separately named, assumed sensitivity variant; never tune the generator to make a chosen model win.

Before fitting, perform a development-only sizing pilot and freeze sample size, seeds and hazard parameters. Do not regenerate final evaluation seeds until a desired event count or result appears. Report retained patients, unique cause-specific events, route-specific events, event-bearing landmark rows, horizon follow-up support and fold composition.

Initial operational gates, explicitly assumed rather than universal statistical guarantees:

- At least 30 unique events for each cause/route estimator used in a deployable bundle.
- At least 20 unique target-event patients and 20 distinct control patients in a reported cause/horizon assessment; at least 50 events for a formal calibration slope.
- At least 100 valid patient bootstrap replicates for release confidence intervals, with at least 80% of requested replicates successful.

Metric gates apply independently. An unsupported AUC or slope need not erase a descriptive Brier score, but all scores with weak event support carry an exploratory status and cannot select a winner. Record `ok`, `insufficient_events`, `insufficient_followup`, `censoring_support_failure` or `fit_failed` with counts and reasons. Report how much of the intended population has valid out-of-fold predictions; never silently compare methods on different surviving subsets.

Replace the existing behavior of assigning zero hazard to a cause with too few events with explicit unsupported-fit status. A release bundle requires supported models for all causes/routes it claims to serve. A missing route must produce an applicability error rather than substitution of the majority route. Quick runs may exercise contracts with relaxed, labelled gates, but cannot promote models or support performance claims.

Acceptance: a cohort with three independent events cannot pass because its landmark table repeats those events; zero events is not interpreted as zero future risk; inadequate folds and missing route models yield explicit reasons; a frozen full evaluation yields enough support or truthfully reports the unmet gate.

## 7. CR-02: observation process and onset sensitivity

Extend synthetic truth with scheduled and attended visit times, last negative echo, first qualifying positive echo, confirmation date, onset interval bounds and latent onset variables. Store this evaluation-only truth separately from predictor inputs. Report five-year attendance among patients alive with the index valve and eligible for that visit; deaths and replacements must not inflate the missing-visit denominator.

Distinguish biological initiation from crossing the haemodynamic endpoint threshold. The existing `svd_onset_date` starts progression; in a gradual scenario it is not necessarily the onset of moderate/severe deterioration. Generate a separate noise-free threshold-crossing time against the patient's reference echo. Measurement noise can create discordant observed positives; report these instead of forcing latent truth inside every observed interval.

Use four prespecified analyses on common patients and frozen splits:

1. Primary: first adjudicated detection, with the existing landmark estimand.
2. Oracle sensitivity: noise-free endpoint-threshold onset, available only in synthetic evaluation. Remove rows at/after oracle onset and report the changed risk set.
3. Interval sensitivity: onset lies in `(last adequate negative, first adjudicated positive]` when that interval is defensible. For eligible pre-interval landmarks, fit/evaluate under left-limit, midpoint and right-end allocations and repeated uniform draws within the interval. Rebuild eligibility for each allocation; do not retain rows already past the allocated event. Report this as an assumption-based interval-imputation sensitivity, not a fitted interval-censored likelihood or formal mathematical bound. Unknown onset before a competing death/replacement requires an explicit missing-onset sensitivity informed by synthetic truth, not automatic SVD-free classification.
4. Visit-process sensitivity: paired cohorts with matched latent patient histories and different regular/informative visit mechanisms. Use separate random streams for latent disease, measurements and attendance so changing visits does not silently change underlying disease. Report patient-balanced evaluation and a separately labelled inverse-visit-intensity analysis fitted within training data, with positivity/weight diagnostics.

Record detection delay, missed threshold crossings, primary-vs-oracle incidence, calibration/Brier changes and how results vary with interval allocations. A genuine interval-censored likelihood model is a separate future extension; this design delivers explicit interval-based sensitivity without claiming to have fitted one. Document this implementation choice in the deviations log.

Acceptance: intervals use only their defined adjudication evidence; latent dates never enter model features; forced visit loss delays detection without changing latent trajectories; no event allocation exceeds death/replacement/follow-up without an explicitly labelled sensitivity assumption; regular versus informative observation results are reproducible.

## 8. CR-03: reliability and applicability

Extend `Reliability` with structured reason codes, applicable scope, family, evaluation support and feature/module exclusions. Retain existing fields for client compatibility. Required reasons include `abrupt_failure_not_reliably_anticipated`, `phenotype_performance_unestablished`, `insufficient_event_support`, `outside_training_followup`, `unknown_device_or_route`, `stale_echo` and `module_unavailable` where applicable.

Do not infer an individual's future regurgitant phenotype from simulator truth or from a scenario name. For a bundle trained only on gradual stenosis, state its limited scope for every patient. Where supported phenotype-specific held-out evidence exists, use it as bundle-level evidence, clearly identified as synthetic. Patient-specific flags may use only contemporaneous clinical findings or documented history. A new regurgitant abnormality triggers the current-finding pathway even if the model predicts low future risk.

Show a concise message beside the numerical output explaining that abrupt failure may lack a preceding gradient signal. Keep the current abnormality, 12-month risk and overdue reminder separate. Compute completeness over trained eligible fields, display excluded modules separately, and do not call an unavailable feature an observed zero. Unsupported predictions fail with a structured reason; supported predictions with limited applicability retain their warning.

Acceptance: changing future echoes, latent phenotype or future event labels cannot change today's prediction or reliability; API and UI preserve warnings for both families; endpoint-met cases still return 409; missing-reference and unsupported-route errors remain distinguishable.

## 9. CR-08: gradient boosting survival comparison

### Estimator and interface

Add `CauseSpecificGradientBoostingModel` using `sksurv.ensemble.GradientBoostingSurvivalAnalysis(loss="coxph")`. This is a boosted survival model with a nonlinear risk function, not three horizon classifiers. The official API provides predicted cumulative hazards for Cox partial-likelihood loss: [estimator reference](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.GradientBoostingSurvivalAnalysis.html).

Fit separate models by route and cause (SVD, death, non-SVD replacement). For each cause, other causes are censored at their occurrence time. This gives each route its own baseline and risk function. Record the additional flexibility relative to Cox's shared coefficients with route-stratified baselines. Preserve explicit landmark time and valve age features. Thrombosis remains a reversible state and a separate secondary outcome, not a fourth absorbing cause in the primary prediction.

Use the same marker eligibility, input feature blocks and training-only imputation/encoding as Cox, but pass numeric features to trees without spline expansion or scaling. Trees learn nonlinearities directly. Record both semantic inputs and transformed columns so comparisons reflect identical information. Freeze a compatible scikit-survival dependency version after checking Python 3.11 wheels on Windows and the Linux container; update the lock and all relevant images. Do not add an unverified version pin in this design.

Refactor the Cox-specific bundle into a model adapter with `fit`, `cumulative_hazards`, `support`, `training_summary` and a separate explanation capability. A bundle stores family, adapter/schema version, preprocessing, route/cause models, time support, integration version and complete configuration. Both adapters produce the same hazard arrays for the shared CIF implementation. Handle legacy `.cox` bundles by explicit migration or rejection; do not silently reinterpret them.

### Tuning and fair evaluation

Run both families across all 11 existing ladder steps and all nine current scenario/variant combinations, plus the defined sensitivities. Primary evaluation uses five outer patient folds. Tune boosting only within three patient-grouped inner folds of each outer training set. Fix a modest initial grid: `n_estimators` in `{100,300}`, `learning_rate` in `{0.03,0.1}`, `max_depth` in `{1,2}`, `min_samples_leaf=20`, `subsample=1.0`, fixed random seeds. These are assumed engineering defaults. Disable internal random-row validation/early stopping, which could split a patient's landmarks across training and validation.

Choose a shared hyperparameter tuple for the complete competing-risk system using mean inner-fold IPCW Brier across the three horizons and three event causes, subject to support and calibration checks. Keep the Cox penalty prespecified as currently configured; if tuned, use the same inner folds. Preprocessing, feature availability and baseline estimation must be refitted within every inner fit.

Save outer out-of-fold probabilities for all four states, fold assignments, training-only decisions, fitted parameter choices, timings and failure reasons. Use paired patient bootstrap differences on the same evaluable patients/landmarks. Report SVD as the primary comparison, with death/replacement/no-event results as consistency and quality checks. AUC alone cannot select a model.

Freeze a final candidate after development evaluation and assess it once on separately seeded synthetic test cohorts. No selection on those final cohorts. Record the exact generator/configuration used; a new seed from the same generator is synthetic replication, not clinical external validation.

### Serving, explanations and promotion

Train and serialize both families. Default prediction family is read from the deployment configuration and recorded in every output. Allow a demo comparison view selecting only published compatible bundles; family selection must not alter eligibility rules, horizons or messages. Display the family and version beside results.

Cox coefficient contributions cannot be reused as boosting explanations. Provide grouped local sensitivity explanations computed by replacing one semantic feature group with a declared training reference and recomputing the 12-month SVD CIF. Preserve internally consistent groups such as current gradient/delta/slope; report direction and change in probability, with method and baseline stated. These are model sensitivities, not treatment effects or causal recommendations. Apply the same explanation method to both families in the comparison view; preserve Cox log-hazard contributions as a separately labelled optional view.

Promotion requires supported fits, passing numerical/leakage tests and a prespecified development comparison. Initial assumed rule: paired mean SVD Brier across horizons improves with a 95% interval excluding zero; no supported horizon worsens by more than 0.005 absolute Brier; absolute probability-scale calibration difference does not worsen by more than 0.01; formal slope remains interpretable. Freeze these tolerances before evaluation. Failure retains Cox as default and publishes the boosting result with its limitations. Passing this gate permits a research-demo default change only; it establishes no clinical validity.

Acceptance: each family survives bundle round-trip with identical probabilities, all causes/routes are accounted for, nested folds never separate a patient's rows, both families use the same final CIF routine and error contract, explanations match the stated quantity, and rejected candidates cannot overwrite the active bundle.

## 10. Implementation map and execution sequence

| Phase | Main files | Deliverable / gate |
|---|---|---|
| A: dataset and availability | `simulation/generators.py`, `simulation/scenarios.py`, `adjudication/framework.py`, `modelling/landmark.py`, `modelling/modules.py`, `modelling/features.py`, scenario/model config | Coherent labels and grade mappings, truth separation, unique-event accounting, training-only eligibility |
| B: probability and evaluation | `modelling/cif.py`, `evaluation/metrics.py`, new `evaluation/support.py`, new `evaluation/sensitivity.py`, `evaluation/ladder.py` | Analytical probability tests, support gates, formal calibration and interval/visit sensitivities |
| C: model family | New `modelling/base.py`, new `modelling/gradient_boosting.py`, existing `cause_specific.py`, `train.py`, `predictor.py`, dependency files | Adapters, supported route/cause fits, nested tuning and versioned bundles |
| D: output and jobs | `extraction/schema.py`, predict service, demo, jobs CLI, plotting, `scripts/build_all.py` | Reliability reasons, family comparison, new result schema and reproducible commands |
| E: validation and documentation | Focused tests, `docs/schemas/`, `docs/deviations.md`, `docs/runbook.md`, model cards | Full scenario report, support failures visible, frozen candidate and test cohort report |

Source paths in the first three rows are relative to `src/kairos/`. Implement phases A–B before interpreting any model-family comparison. Add tests to the existing suites where they exercise the same contract; create separate suites for boosting, support gates and observation sensitivity.

Proposed CLI extensions (to implement): `train --family cox|gradient_boosting`, `evaluate --families cox gradient_boosting --all`, and `evaluate --sensitivity observation`. `build_all.py --quick` runs both families on small cohorts with limited tuning and clearly marks outputs as smoke tests. The full command runs the frozen settings. Avoid forcing the entire expensive comparison into every unit-test invocation; use a dedicated full evaluation job with checkpoints.

Keep artifacts under immutable run identifiers, then publish family-specific pointers only after successful completion. Separate `quick` and `full` namespaces. Record cohort, feature, model, integration and evaluation hashes; metrics from different endpoint or integration versions must not be merged into one ladder. Retain the previously active bundle for rollback. Measure job memory and runtime before changing Azure job resources; scale only if the expanded workload requires it.

Update deviation entries 7 (calibration), 9 (endpoint consistency), 14 (unconverted Lp(a) effect) and 16 (availability) to distinguish historical behavior from completed changes. As part of truthful scenario provenance, an unconverted sub-distribution effect used in a cause-specific generator must be labelled an assumed sensitivity parameter, not a validated conversion. Record independent synthetic replication and any remaining interval-imputation assumptions explicitly.

## 11. Release evidence and boundaries

Completion requires the existing tests plus the targeted contracts above; clean lint, privacy and schema checks; and a full report covering all families, ladder steps, scenario variants, horizons, support statuses, paired differences and sensitivity results. Include independent patient/event counts, eligibility exclusions, sample-size settings, bootstrap validity and fit failures. Replace old calibration conclusions only after regenerating metrics under the new definitions.

Azure deployment/runtime acceptance and the hosted extraction comparison remain separate unverified items from the earlier review. This design does not claim they are complete or require live infrastructure changes to accept the local modelling work. If the revised image is subsequently deployed, verify loading both model families, service error/reliability contracts, job artifact persistence and the authenticated demonstration, and record actual results in the runbook.

The design is complete when its implementation can be assessed against these gates. The modelling change set is complete only when implementation and full evaluation evidence exist; creating this document does not close the review findings.
