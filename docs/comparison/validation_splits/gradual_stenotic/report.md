# Comparator ladder and validation splits: `gradual_stenotic`

Generator 2.2, seed 20260916. SVD at 5 years, pooled landmark rows. Synthetic and illustrative.

## Comparator ladder (five-fold, patient-grouped; 10701 of 10701 rows scored by every comparator)

| comparator | n | n_events | observed | mean_predicted | brier | ipa | auc | cal_slope_logit |
|---|---|---|---|---|---|---|---|---|
| R1_valve_age_and_type | 10701 | 329 | 0.0412 | 0.0439 | 0.0386 | 0.0237 | 0.6909 | 1.0029 |
| R2_pre_implant | 10701 | 329 | 0.0412 | 0.0449 | 0.0373 | 0.0555 | 0.6816 | 1.0593 |
| R3_reference_echo_once | 10701 | 329 | 0.0412 | 0.045 | 0.0374 | 0.0526 | 0.675 | 1.0111 |
| R4_current_gradient_and_change | 10701 | 329 | 0.0412 | 0.0468 | 0.0372 | 0.0596 | 0.7131 | 1.0501 |
| KAIROS_core | 10701 | 329 | 0.0412 | 0.0484 | 0.0354 | 0.1051 | 0.7328 | 1.038 |
| KAIROS_surveillance_blinded | 10701 | 329 | 0.0412 | 0.0484 | 0.0354 | 0.1051 | 0.7328 | 1.038 |
| KAIROS_visit_history_blinded | 10701 | 329 | 0.0412 | 0.0483 | 0.0354 | 0.1052 | 0.7328 | 1.0403 |

Paired IPCW Brier differences, patient bootstrap (200 replicates); negative favours the first:

- KAIROS_core vs R4_current_gradient_and_change: -0.0018 (95% CI -0.0030 to -0.0007)
- KAIROS_surveillance_blinded vs R4_current_gradient_and_change: -0.0018 (95% CI -0.0030 to -0.0007)
- KAIROS_visit_history_blinded vs R4_current_gradient_and_change: -0.0018 (95% CI -0.0030 to -0.0007)
- R3_reference_echo_once vs R4_current_gradient_and_change: +0.0003 (95% CI -0.0008 to +0.0014)
- KAIROS_core vs R1_valve_age_and_type: -0.0032 (95% CI -0.0045 to -0.0018)

## Temporal split: fitted on implants up to 2016, scored on later implants

1284 training patients, 1106 test patients.

| comparator | n | n_events | observed | mean_predicted | brier | ipa | auc | cal_slope_logit |
|---|---|---|---|---|---|---|---|---|
| R1_valve_age_and_type | 4814 | 142 | 0.0409 | 0.0436 | 0.038 | 0.0308 | 0.7446 | 1.3283 |
| R4_current_gradient_and_change | 4814 | 142 | 0.0409 | 0.046 | 0.0366 | 0.0678 | 0.7319 | 1.1583 |
| KAIROS_core | 4814 | 142 | 0.0409 | 0.0546 | 0.0337 | 0.1423 | 0.7806 | 1.1842 |

## Leave one design class out

| model | n | n_events | observed | mean_predicted | brier | ipa | auc | cal_slope_logit |
|---|---|---|---|---|---|---|---|---|
| externally mounted pericardial / R1_valve_age_and_type | 1382 | 118 | 0.1124 | 0.0479 | 0.1037 | -0.0392 | 0.7378 | 2.1927 |
| externally mounted pericardial / KAIROS_core | 1382 | 118 | 0.1124 | 0.0699 | 0.0921 | 0.0763 | 0.7316 | 1.0294 |
| stented bovine pericardial, internally mounted / R1_valve_age_and_type | 3370 | 92 | 0.0378 | 0.0943 | 0.0383 | -0.051 | 0.7437 | 2.2673 |
| stented bovine pericardial, internally mounted / KAIROS_core | 3370 | 92 | 0.0378 | 0.0887 | 0.036 | 0.0112 | 0.7349 | 1.1507 |

Not scored (fewer than 10 event patients in the class): balloon-expandable intra-annular TAVR (9), mechanically expanded intra-annular TAVR (0), self-expanding intra-annular TAVR (1), self-expanding supra-annular TAVR (6), stented porcine (8), stentless (2), sutureless or rapid-deployment (6).
