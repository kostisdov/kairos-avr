# Paired family comparison: penalised Cox versus gradient boosting

Scenario `gradual_stenotic`, synthetic. Out-of-fold predictions on the identical rows for both
families, 200 bootstrap replicates resampled by patient. Produced by
`services/jobs/cli.py compare`; the raw decisions are the JSON files beside this note.

The promotion rule was pre-specified: gradient boosting replaces the penalised Cox model only if the
mean structural-deterioration Brier score improves with a 95 percent interval below zero, and no
horizon worsens beyond tolerance. **The first criterion failed at every ladder step.**

| Ladder step | Decision | Mean Brier difference | 95% interval | Rows | Patients |
|---|---|---|---|---|---|
| reference | **retain_cox** | +3.42e-06 | [-6.44e-05, +6.45e-05] | 10,701 | 2,390 |
| core | **retain_cox** | +2.04e-04 | [-2.94e-04, +6.96e-04] | 10,701 | 2,390 |
| core_plus_both | **retain_cox** | +5.65e-04 | [-3.97e-06, +1.09e-03] | 10,701 | 2,390 |

A positive difference means gradient boosting scored worse. Every interval spans zero, so the two
families are indistinguishable on this scenario and the challenger did not earn promotion. The
remaining criteria, on one- and three-year Brier and on absolute observed-minus-predicted, all
passed, so boosting is not *worse* beyond tolerance either; it simply adds nothing.

The result is published rather than dropped. A challenger that fails its own promotion rule is
evidence that the rule was applied, not evidence of a weak submission.
