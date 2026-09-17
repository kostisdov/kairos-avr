# Model card: cause-specific Cox family (`cox`)

Status: synthetic scenarios only. Illustrative and not validated. Not for clinical use.

## What it predicts

For a bioprosthetic aortic valve (SAVR or TAVR) at a landmark prediction time, this family predicts
four competing-risk probabilities at 1, 3 and 5 years: SVD before death, death before SVD, alive with
the valve intact, and SVD within 12 months. Each cause (SVD and death) gets its own Cox model. The
coefficients are shared across valve routes, with a separate baseline hazard for each route. The
cumulative incidences come from the Breslow cumulative hazards.

## Scope

- Training data: synthetic cohorts from the KAIROS scenario generator only. No real patient data went into fitting.
- Endpoint: gradual haemodynamic structural valve deterioration (stenosis), found on serial echoes and adjudicated as the generator defines it. Abrupt failure (for example leaflet tear or thrombosis) is outside the model. The service shows the abrupt-failure message next to every output.
- Numbers from synthetic replication (new seeds, variants, final-test seeds) are internal replication, not external validation (deviation 44).

## Support

Support is counted in unique event patients for each route and cause (CR-04). The thresholds are
`fit_min_unique_events` and `baseline_min_unique_events` in `config/model.yaml`.

- `ok`: the covariate model was fitted.
- `reduced_baseline_only`: only the covariate-free route-stratified Nelson-Aalen baseline was fitted. The output says so and never calls it a covariate model.
- `insufficient_events` or an unseen route: the model refuses to predict (`UnsupportedFitError` / `UnsupportedRouteError`). It never returns zero hazard or swaps in another route.

Each bundle's card (`models/runs/{run_id}/cox/card.json`) records `support_mode` and `model_support`. `train` prints them too.

## Integration version

`pch-expm1/1` (`kairos.modelling.cif.INTEGRATION_VERSION`). The survival curves and cumulative
incidences integrate Breslow hazards with a piecewise-constant-hazard convention (deviation 39).
Bundles with a different integration version are rejected, not reinterpreted.

## Explanation method

`grouped_local_sensitivity` (`kairos.modelling.explain`). For one patient, each feature group is set
to a reference value, and the change in 12-month SVD probability is reported:

- For serial-echo groups, the reference is the patient's own reference echo.
- For static, biomarker and exposure groups, it is the training median or mode.

These are model sensitivities, not treatment effects. The gradient-boosting family is explained the same way.

## Limits

- The data are synthetic, so performance reflects the generator's assumptions, not real patients.
- Proportional hazards within each cause is assumed. Coefficients are shared across routes (deviation 41).
- Uncertain SVD candidates are censored at the candidate date in the primary analysis (deviation 43).
- Onset timing depends on the visit schedule. The interval-imputation sensitivity bounds that effect, but it is not a likelihood-based correction (deviation 40).
- Hyperparameters and gates come from development runs. Quick-namespace bundles are stamped `smoke test: not evidence`.
- Serving goes through `models/{namespace}/cox/active.json` (`publish` / `publish --rollback`). The service checks the SHA-256 before it loads a bundle.
