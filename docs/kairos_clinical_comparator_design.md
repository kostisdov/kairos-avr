# Clinical echo comparator and Action 1: implementation design

Date: 17 September 2026. Status: proposed design; no runtime, endpoint, fitted model or threshold changed by this document.

Scope correction from the owner: `kairos_varc3_sol_implementation_design.md` is the controlling Sol brief and specifies an isolated comparator with read-only access to existing predictions and labels, plus the explicitly requested Patient Summary tab and Azure OpenAI/Foundry findings draft. The proposed engine, prediction API, bundle and action-threshold integrations below are deferred; use this document's evaluation methods only where compatible with that brief. Additive summary UI/settings are allowed. Do not change the current engine, relabel data, regenerate cohorts or retrain models for this task.

The requested change is justified. Keep the patient's reference echocardiogram and add the current-echo gradient rule to the evaluation ladder. Retain valve age and type as a minimal statistical comparator. Clinical incremental value must be evaluated against an explicit decision strategy, on the same eligible patients and prediction times, with twelve-month outcomes.

## 1. Terminology and claim

| Term | Meaning | Decision |
|---|---|---|
| Patient reference echocardiogram | First adequate study 30–180 days after implantation | Preserve the established selection and adequacy rules |
| Minimal statistical comparator | Current `reference` model: valve age, route and design class | Retain internal ID for artifact compatibility; relabel displays |
| Clinical gradient trigger | Current mean gradient ≥20 mmHg **and** increase from the patient's reference ≥10 mmHg | Add fixed, unfitted comparator `echo_gradient_rule_v1` |
| KAIROS model | Twelve-month cumulative incidence of adjudicated SVD before death or non-SVD replacement | Evaluate added prognostic and decision value |

The rule is a clinically relevant **gradient screening trigger**, not a complete representation of cardiologists' assessment. VARC-3 and ASE criteria also consider EOA/DVI changes and intraprosthetic regurgitation; a negative gradient trigger does not establish normal valve function. Do not call it the complete standard of care or say “nothing wrong.” [VARC-3](https://www.jacc.org/doi/10.1016/j.jacc.2021.02.038), [ASE 2024](https://www.asecho.org/wp-content/uploads/2024/01/PIIS0894731723005333.pdf).

The project's 30–180-day reference window remains a protocol choice. These sources describe a 1–3-month reference assessment; adding the comparator does not silently alter the settled project window or claim they are identical.

Panel wording: “The current gradient trigger is not met. KAIROS estimates a twelve-month SVD risk of X%. We evaluate whether acting on that additional warning improves net benefit over the gradient trigger.” X must come from the actual model output. A patient example illustrates discordance; cohort evaluation establishes whether that discordance is useful.

## 2. Findings in the current implementation

| Location | Current behavior | Required addition |
|---|---|---|
| `adjudication/framework.py::select_reference` | Selects the first adequate echo in the configured window | Reuse unchanged |
| `modelling/landmark.py::echo_features` | Produces `ref_gradient`, `current_gradient`, `delta_gradient`, echo age and stage | Capture rule result from the same raw echo pair before imputation |
| `modelling/modules.py::ladder_steps` and `config/model.yaml` | Every ladder entry supplies feature blocks for a fitted model | Keep training ladder; add a separate comparator registry and a combined report |
| `evaluation/ladder.py::evaluate_ladder` | Produces patient-grouped out-of-fold probabilities and coverage | Attach paired comparator results using stable landmark identifiers |
| `evaluation/metrics.py` | Has competing-risk outcome coding, IPCW and patient bootstrap; no net-benefit evaluator | Reuse primitives in a dedicated decision-analysis module |
| `evaluation/compare.py` | Compares boosting with Cox and records family promotion | Keep family selection separate from the Action 1 threshold policy |
| `modelling/predictor.py::Predictor` | Reads an assumed 0.05 twelve-month threshold from live config | Record policy provenance and eventually load an evaluated policy from the bundle |
| `extraction/schema.py`, `services/demo/app.py` | Exposes current abnormality, earlier assessment and overdue surveillance | Add explicit comparator and threshold-policy fields beside existing outputs |

The Boolean predicate is small. Valid comparison, endpoint timing, missingness and policy provenance constitute the larger implementation.

## 3. Rule contract

Add `src/kairos/comparators/echo_gradient.py`, with an immutable `GradientRuleResult` and a pure entry point:

```python
evaluate_gradient_rule(reference, current, prediction_time, stale_months=18)
# -> GradientRuleResult
```

The wrapper must receive the same selected reference and current prosthetic echo used for model features. It must not independently choose a more favorable reference. Inputs come from the same index valve and have dates on or before prediction time. Reuse the existing time-zero selection; a gradient-only study must not become a new qualifying reference under this change.

Result contract:

```text
comparator_id: echo_gradient_rule_v1
status: positive | negative | not_evaluable
triggered: true | false | null
reference_date, current_echo_date, prediction_time
reference_gradient_mmhg, current_gradient_mmhg, delta_gradient_mmhg
reason: threshold_met | threshold_not_met | no_reference |
        missing_gradient | invalid_gradient | stale_echo |
        invalid_chronology | incompatible_index_valve
```

Implementation rules:

1. Use finite, nonnegative measured mean gradients in normalized mmHg. Missing/NaN/infinite/invalid values produce `not_evaluable`, never an imputed negative. Valid extreme values retain upstream source-verification flags; this rule introduces no arbitrary upper cutoff.
2. Compute `delta = current - reference` directly. Trigger exactly when `delta >= 10.0 and current >= 20.0`; do not round before comparison, use OR, compare with the previous follow-up echo, or introduce an epsilon tolerance.
3. Use the latest selected prosthetic study. If its gradient is missing, return unavailable rather than silently searching backwards for a usable value. Sort consistently with the existing `(date, index)` ordering; unresolved duplicate-study conflicts are a data-quality issue upstream.
4. Reuse the existing 18-month freshness boundary. Stale inputs return `not_evaluable` for the current decision comparison; retain measured values and dates for display. At echo landmarks the current study is contemporaneous, so this mainly affects live requests between visits.
5. At the reference study itself, the delta is zero and the trigger is negative when inputs are valid. Identify reference-only rows in reporting because they cannot provide serial warning.
6. EOA, DVI, regurgitation and later adjudication are not inputs to this gradient predicate. The existing full staging and eligibility logic remains separate and visible.

Store comparator values as audit data, outside fitted feature blocks. Extend `AUDIT_COLUMNS`/the predictor exclusion guard for any new audit columns placed on the landmark table. Do not accidentally fit the rule status, future confirmation, or outcome information as a new feature.

## 4. Population and timing: the critical comparison boundary

Primary question: among the existing eligible landmarks, does today's rule or model better identify **future adjudicated SVD within twelve months**?

Use the unchanged primary label policy and risk set. All methods use the same patient, index valve, landmark, reference study, twelve-month horizon and outcome. Death and non-SVD replacement are competing events. No SVD endpoint-establishing echo may become a predictor of that endpoint. Keep exclusions/censoring for unresolved candidates and existing endpoint/replacement handling.

This has a consequential limitation: a gradient-positive echo with corroborating EOA/DVI changes can already establish the endpoint or initiate an unresolved candidate. The current pipeline removes that landmark from primary prognostic evaluation. Thus the gradient comparator may have very few positives, particularly in stenotic scenarios. This is a property of the estimand, not proof of model superiority over current diagnosis.

Before computing gains, publish an attrition audit: intended rows/patients, endpoint and unresolved-candidate exclusions, rule-positive/negative/unavailable counts on eligible rows, and reference-only versus later rows. Report which excluded endpoint/candidate echoes met the gradient trigger in a separate descriptive audit, never as prognostic predictors.

If the eligible rule has zero positives, its net benefit is zero and equals assess-none. Emit `degenerate_no_rule_positives` and explicitly state that the comparison tests earlier risk stratification before the trigger. Do not relax endpoint exclusions to improve the comparison. A comparison of complete clinical surveillance pathways would require a separate design and observed clinician decisions.

Report all scenarios, including abrupt regurgitant deterioration, where a gradient-only rule has limited scope. Show stenotic, regurgitant and mixed results when supported. Synthetic latent phenotype can label evaluation strata only; it must not enter model or rule inputs. Reuse existing observation-process sensitivities because future detection depends on visit timing.

## 5. Evaluation outputs and net benefit

Add `src/kairos/evaluation/decision_curve.py`. Public functions should cover `evaluate_decision_strategies`, `paired_net_benefit` and `summarise_rule_negative_patients`; these consume raw rule results and saved out-of-fold model outputs, not fitted probabilities on training rows.

Join on `(patient_id, landmark_date)` within the scenario/index-valve context. Assert uniqueness, matching fold, matching time/event and identical row keys; never rely only on equal lengths or patient ordering. Compare on the intersection of evaluable rule rows and supported finite model predictions. Report model-only, rule-only and neither coverage separately. Recompute each model's paired comparisons on its own common set and publish an all-method common-set view for ranking.

At horizon `tau = 1.0` year, reuse `outcome(..., state='svd')` and `ipcw_weights`. SVD by the horizon has `Y=1`; observed competing events and follow-up without SVD through the horizon have `Y=0`. Early censoring contributes zero IPCW outcome weight and is not counted as a known negative.

For decision `a_i`, population weight `q_i`, IPCW weight `w_i` and threshold preference `p_t`:

```text
NB(a, p_t) = sum[q_i * w_i * a_i *
                    (Y_i - (1 - Y_i) * p_t / (1 - p_t))] / sum[q_i]
```

Use `q=1` for the existing pooled-landmark primary analysis. Also report patient-balanced results, recalculating `q=1/(number of included landmarks for that patient)` on the paired set. The denominator includes all paired eligible decision opportunities, including early-censored rows; it is not the sum of IPCW weights. Reuse existing censoring-survival and bootstrap support gates and publish effective sample size and weight diagnostics. This implementation extends the existing outcome/IPCW machinery; it does not imply that DCA establishes a causal effect of earlier assessment. [Competing-risk validation methods](https://www.bmj.com/content/377/bmj-2021-069249).

Evaluate these strategies at each prespecified `0 < p_t < 1`:

| Strategy | Action indicator |
|---|---|
| Gradient trigger | `rule.triggered` |
| KAIROS | `p_svd_12m >= p_t` |
| Trigger plus KAIROS | `rule.triggered OR p_svd_12m >= p_t` |
| Assess all | Always true |
| Assess none | Always false |

The combined strategy answers the practical question of adding a model flag while retaining the gradient trigger. Report its gain over the trigger alongside the standalone model comparison. Existing broader clinical abnormality handling remains a separate care pathway, outside this simplified strategy definition.

The rule's classification is fixed across thresholds, but its net benefit changes with the false-positive penalty. Never encode its positive/negative output as a 100%/0% future SVD probability. Calibration, Brier score and probability-based AUC are `not_applicable` for this comparator. The combined ladder report contains a rule row with coverage, alert rate, twelve-month sensitivity/specificity/PPV and net benefit; probabilistic models retain their existing calibration and Brier columns.

Bootstrap patients using the same resampled clusters for every strategy. Re-estimate the evaluation censoring distribution within resamples; reuse the same resample weights across strategies. Publish paired 95% intervals for incremental net benefit and successful/failed resample counts. These intervals are conditional on saved out-of-fold predictions, not uncertainty from refitting the entire development pipeline. Preserve descriptive outputs with exploratory/support labels where gates fail, consistent with current repository policy.

Also export the four rule/model alert groups, alert rates per 100 decision opportunities, distinct alerted patients, additional future SVD cases identified, and weighted twelve-month risk in the rule-negative/model-positive group. Any lead-time summary uses pre-endpoint alert dates, retains never-alerted patients in the denominator, and is secondary; do not summarize only successfully detected cases.

## 6. Action 1 and threshold policy

Action 1 is an earlier-assessment flag. It is not a recommendation for reintervention. Keep its operational meaning and existing current-abnormality handling.

Do not select a threshold merely because two empirical curves cross. In decision-curve analysis, the threshold also expresses the trade-off between unnecessary assessment and missed future SVD. Different thresholds encode different preferences; maximizing net benefit across those thresholds is not an unconstrained statistical optimization problem. [Net-benefit methods](https://www.bmj.com/content/352/bmj.i6).

For the first implementation:

1. Evaluate the existing **5% assumed threshold** as the prespecified prototype operating point. Plot an exploratory grid from 1% to 20% in 1-percentage-point increments, explicitly an engineering assumption rather than a clinical recommendation.
2. Report the supported threshold range where KAIROS and the combined strategy improve net benefit over the rule. At 5%, also compare with assess-all and assess-none. Show calibration and assessment burden beside net benefit.
3. Record separate status for the estimate, uncertainty and evidence provenance. A positive point estimate alone is not a demonstrated gain. A proposed research evidence gate is a paired 95% lower bound above zero versus the rule, positive net benefit, no point-estimate inferiority to assess-all, and supported twelve-month calibration/count diagnostics. This gate is an explicit proposed convention, not a guideline.
4. If no threshold is supported, publish that result and keep the 5% setting labelled assumed; do not tune until a favorable claim appears. Synthetic runs still complete and produce exploratory outputs. A clinical operating threshold remains unvalidated.
5. If a different cutoff is subsequently selected using development results, freeze the clinically acceptable threshold range, burden constraints, decision criterion and tie-break before selection. Select within outer-training data using inner out-of-fold predictions; outer-held-out outcomes cannot choose their own cutoff. Evaluate the final frozen policy once in an independent test cohort. A new synthetic seed is synthetic replication, not external clinical validation.

Add a `decision_policy` artifact containing policy version, comparator version, action, horizon, operating threshold, threshold grid, candidate model/step/family/version, cohort and label-policy hashes, weighting, common-row counts, criterion, net-benefit results, evidence status and timestamp. Include the complete policy in the candidate freeze and model card. A configuration edit must not silently substitute a different threshold for a frozen evaluated policy.

For legacy bundles without the artifact, preserve the existing demo threshold but return `threshold_source=legacy_assumed_config`. The family-promotion rule remains intact; passing Cox-versus-boosting promotion does not establish benefit over the clinical trigger.

## 7. Configuration, API and display

Keep `config/model.yaml::ladder` exclusively for fitted feature-block models. Add a sibling `clinical_comparators` registry and a `decision_analysis` section with rule ID, 12-month horizon, 5% prototype operating point, grid and strategy list. Add matching plan fields and hashes in `config/evaluation_plan.yaml`. The human-facing ladder merges fitted models and decision comparators using an explicit `result_kind`; training and calibration plots filter by that field.

Extend `Prediction` with optional, backward-compatible `clinical_comparator` and `decision_policy` objects. The live result includes raw gradients/dates, trigger status and reason, actual `p_svd_12m`, operating threshold, model flag, rule/model discordance and policy evidence label. Evaluate the comparator before feature imputation. Unavailable comparison does not automatically suppress an otherwise eligible model prediction. Preserve existing errors for absent qualifying reference or endpoint ineligibility; do not manufacture a risk result for those cases.

Use the same pure evaluator during landmark construction and live prediction. Existing `current_abnormality`, `earlier_assessment` and `overdue_surveillance` fields retain their meanings; a negative gradient result never clears another abnormality flag.

Display specimen for a synthetic fixture with reference 11 and current 18 mmHg:

```text
Gradient trigger: not met
Mean gradient: 18 mmHg; increase from reference: 7 mmHg
KAIROS twelve-month SVD risk: [actual model output]%
Earlier-assessment threshold: 5% — assumed prototype setting
[Show model flag only if the actual risk is at least the threshold]
Synthetic, illustrative and unvalidated
```

Do not hand-pick or invent a prediction to produce discordance. The reproducible demo fixture may be selected from synthetic evaluation outputs, with its selection method recorded. Show current staging separately, particularly for regurgitant abnormalities and unresolved mechanisms.

## 8. Delivery sequence and acceptance

| Work package | Files and integration | Acceptance |
|---|---|---|
| A. Shared rule | New comparator package; landmark audit adapter | One rule result from the same raw pair in batch and live use; no feature-block changes |
| B. Paired decisions | New decision-curve module; `evaluation/ladder.py`, `support.py`, `services/jobs/cli.py` | Versioned rule audit, strategy curves, paired deltas, discordance and coverage artifacts on identical rows |
| C. Policy provenance | Config/plan; predictor bundle/card; candidate-freeze path | 5% evaluated but labelled assumed; policy reproducible and no automatic promotion from a crossing |
| D. Presentation | Schema exports, predictor, demo, plots, ladder summary | Rule visible in combined ladder; N/A probability metrics; actual model risk shown beside rule status |
| E. Documentation | Proposal, change log, detailed design, runbook | Two baselines named distinctly; usefulness claims tied to comparator results and evidence level |

Proposed run artifacts under `metrics/runs/<run_id>/<scenario>/<variant>/clinical_comparison/`: `rule_audit.parquet`, `decision_curves.csv`, `paired_differences.csv`, `discordance.csv`, `coverage.json`, `policy.json`. Match existing storage/privacy conventions: patient identifiers and row-level data stay in permitted local/private outputs; published reports contain aggregates only.

Required meaningful tests:

- Boundary truth table: reference/current 10/20 and 15/25 positive; 5/19, 15/20 and 10/19.999 negative. Missing/nonfinite/negative values unavailable. Same reference/current zero delta negative. Chronology, freshness and index-valve guards covered.
- Changing only a later echo or future confirmation cannot change a prior rule result or model input. Endpoint-establishing rows remain excluded. Audit data cannot enter a fitted feature block.
- Raw missing gradients stay unavailable even when the feature pipeline imputes gradients. Batch/live parity includes duplicate ordering and prediction-time filtering.
- Hand-calculated no-censoring decision table verifies net benefit, assess-none zero and assess-all identity. An identical model/rule decision vector gives exactly zero paired difference. A zero-positive rule is labelled degenerate, not reported as a fitting failure.
- Death/replacement before twelve months are known competing outcomes; early censoring is not a true negative; follow-up ending exactly at the horizon is known. Unsupported weights produce explicit status.
- Shuffled keys align correctly; duplicate keys, mismatched outcomes or folds fail. Bootstrap resampling keeps each patient together and pairs the strategies. Patient-balanced weights recompute after common-set filtering.
- Rule probability metrics serialize as N/A with a reason. Schema export and demo rendering handle all three statuses and legacy bundle threshold provenance. Existing endpoint errors and broader abnormality flags still work.
- Changing held-out outcomes cannot change an operating threshold selected without them. Failed evidence checks cannot create an “evaluated clinical threshold” label.

Run these focused tests plus the existing landmark leakage, CIF metrics, prediction service and schema suites. Run quick scenario evaluation for integration; use full planned evaluation for any supported synthetic comparison claim. Do not regenerate models or change deployment merely to add an unfitted rule.

## 9. Proposed replacement text for the draft

“The patient's reference echocardiogram is the first adequate study 30–180 days after implantation and remains the reference for subsequent change. Evaluation includes two distinct comparators: a minimal statistical model using valve age and type, and a fixed clinical gradient trigger requiring an increase of at least 10 mmHg to a current mean gradient of at least 20 mmHg. The latter is a screening comparator, not the complete assessment of prosthetic valve dysfunction. At matched eligible landmarks, we compare twelve-month decisions using KAIROS, the trigger, and their combination, with competing-risk net benefit, calibration of model probabilities, assessment burden and paired uncertainty. The earlier-assessment threshold is prespecified and evaluated against the trigger; the prototype's 5% setting remains assumed until supported evaluation and clinical justification establish an operating policy. Rule-negative/model-positive cases illustrate potential advance warning and are assessed as a group for observed outcomes. All present model results are synthetic and unvalidated.”
