# Model card: gradient-boosting family (`gradient_boosting`)

Status: synthetic scenarios only. Illustrative and not validated. Not for clinical use. This is a
challenger to the Cox family. It is served by default only after a comparison decision and a frozen
final test.

## What it predicts

The outputs match the Cox family: four competing-risk probabilities at 1, 3 and 5 years (SVD before
death, death before SVD, alive with the valve intact, SVD within 12 months) at a landmark prediction
time. Each cause gets a boosted log-risk function, fitted separately for each valve route. The
cumulative hazards use a Breslow baseline, and they combine into cumulative incidences with the same
integration code the Cox family uses.

## Scope

- Training data: synthetic cohorts from the KAIROS scenario generator only.
- Endpoint: gradual haemodynamic structural valve deterioration (stenosis). Abrupt failure is outside the model.
- Synthetic replication is not external validation (deviation 44).

## Support

The support rules match the Cox family and are applied per route (CR-04):

- `ok`: a boosted model was fitted.
- `reduced_baseline_only`: only the baseline was fitted.
- `insufficient_events` or an unseen route: the model refuses to predict. It does not extrapolate.

Because risk functions are fitted per route, a route with little data gets no strength borrowed from other routes (deviation 41). Each bundle's card records `support_mode` and `model_support`, and `train` prints them.

## Integration version

`pch-expm1/1`, the same as the Cox family (deviation 39). The predict service offers
`?family=gradient_boosting` only when this bundle's endpoint version, horizons and integration
version all match the default family's.

## Explanation method

`grouped_local_sensitivity`, the same method as the Cox family. For one patient, each feature group
is set to a reference value (the patient's own reference echo, or the training median or mode), and
the change in 12-month SVD probability is reported. There are no SHAP values or split-based
importances. These are model sensitivities, not treatment effects.

## Limits

- The data are synthetic, and the model is more flexible than Cox. On small cohorts it can fit generator noise, so tuning is nested within training folds and patient-grouped.
- The tuning objective is the unscaled mean Brier score (deviation 42). Discrimination and calibration slope are reported but not optimised.
- Coefficients are not interpretable, so explanations are local and grouped only.
- Uncertain candidates are censored at the candidate date in the primary analysis (deviation 43). Onset timing is handled by interval-imputation sensitivity (deviation 40).
- Quick-namespace bundles use `quick_grid` and relaxed gates, and are stamped `smoke test: not evidence`.
- Serving goes through `models/{namespace}/gradient_boosting/active.json` (`publish` / `publish --rollback`). The service checks the SHA-256.
