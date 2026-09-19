# KAIROS standalone comparator and Patient Summary implementation report

Date: 17 September 2026

## Implemented scope

- Added the independent `varc3_hvd_comparator_v1` predicate and a read-only adapter. It uses the selected KAIROS 30–180-day reference pair but does not call `stage_hvd`, fit a model, impute missing values, or assign an SVD mechanism.
- Added keyed pairing, IPCW decision metrics, pooled/patient-balanced net benefit, paired patient bootstrap support, coverage diagnostics, input hashes, and comparison-only artifact/report writers.
- Added the standalone `scripts/evaluate_clinical_comparator.py` command and `config/clinical_comparison.yaml`. The command requires caller-specified saved landmarks, held-out predictions, and selected comparator pairs; it does not build a cohort or train a model.
- Added strict Patient Summary fact/draft schemas, deterministic comparison wording and fallback, session-scoped snapshot binding, safe Markdown export, and an isolated Azure OpenAI/Foundry writer.
- Added explicit summary deployment/reasoning/token settings. An empty `KAIROS_OPENAI_SUMMARY_DEPLOYMENT` leaves deterministic summary functionality available and disables hosted narrative generation.
- Added the first `Patient Summary` tab, one common model-family toolbar, explicit family-comparison execution, partial-failure presentation, version/family identity checks, shared structured reliability rendering, and local comparator reuse.

The implementation does not edit `src/kairos/varc3.py`, adjudication, endpoint labels, features, fitting, model configuration, bundles, prediction schemas, prediction endpoints, thresholds, or active deployment pointers.

## Verification

- Focused comparator/evaluation/summary/UI/service regression set: **55 passed**.
- Full repository suite: **281 passed, 1 timing-only failure** on the first run. The failed existing leakage test compared two otherwise identical fit dictionaries whose recorded `seconds` fields were 1.1 and 0.9. Its isolated rerun passed (**1 passed**).
- Ruff checks for all changed Python modules and tests: **passed**.
- No Azure OpenAI call was made. Narrative behavior was tested with fake structured-output clients, including real-source refusal, explicit deployment and sampling settings, fabricated numeric content, invalid evidence/actions, and deterministic fallback behavior.
- No comparator evaluation run was launched against an arbitrarily selected historical run. The command reports missing/incompatible input rather than regenerating a cohort or fitting predictions.

## Genuine two-family check

The prior `test_second_compatible_family_is_served` remains a routing/metadata test because it copies a Cox bundle and changes only its family label.

A separate read-only isolated service check used the saved compatible bundles under `artifacts/models/0.1.0_src-d17fe77e0602/`:

| Family | Verified adapter class | Response identity | Twelve-month risk on the same synthetic request |
|---|---|---|---|
| Cox | `CauseSpecificCoxModel` | `cox`, `0.1.0+src-d17fe77e0602` | 2.56% |
| Gradient boosting | `CauseSpecificGradientBoostingModel` | `gradient_boosting`, `0.1.0+src-d17fe77e0602` | 1.87% |

Both were listed compatible by the isolated service, shared integration `pch-expm1/1`, schema 2, ladder step `core_plus_both`, and the same scenario set. This verifies genuine adapters and paired request routing without changing active pointers. It remains synthetic, illustrative and unvalidated; the patient-level arithmetic difference is not evidence that either family is superior.

A paired Streamlit AppTest then loaded those bundles through temporary `full/<family>/active.json` pointers. It verified both toolbar options, explicit family comparison, the permitted arithmetic delta, switching the shared selector to gradient boosting, and a gradient-boosting-bound Patient Summary. This check exposed and fixed a demo-only gap: in-process mode had loaded only the legacy `latest` bundle rather than the service's active multi-family set. No persisted active pointer was changed.

## Operational items not asserted

- Hosted summary generation is not operational until an administrator explicitly configures a reachable `KAIROS_OPENAI_SUMMARY_DEPLOYMENT` and runs a synthetic connectivity check.
- A clinical-comparison report requires a deliberately selected compatible saved landmark/outcome table, saved out-of-fold predictions, and selected reference/current pair table. No source run was silently chosen.
- The implementation does not claim external VARC-3 validation, diagnostic superiority, calibrated net benefit, or a clinically justified threshold.
