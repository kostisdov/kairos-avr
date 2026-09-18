# KAIROS against current practice: `regurgitant_abrupt`

Synthetic and illustrative. Out-of-fold throughout: no patient is scored or guided by a model trained on them.

## 1. The model as a risk score (12-month SVD, landmark level)

| population | score | n | n_events | observed | mean_predicted | brier | ipa | auc | cal_slope_logit |
|---|---|---|---|---|---|---|---|---|---|
| all_landmarks | model | 10422 | 65.0 | 0.0066 | 0.0066 | 0.0065 | 0.0085 | 0.7167 | 1.1495 |
| all_landmarks | reference | 10422 | 65.0 | 0.0066 | 0.0066 | 0.0065 | 0.004 | 0.6589 | 1.1057 |
| all_landmarks | calendar | 10422 | 65.0 | 0.0066 | 0.0066 | 0.0066 | 0.001 | 0.582 | 0.7974 |
| primary_population | model | 10380 | 62.0 | 0.0063 | 0.0065 | 0.0062 | 0.0074 | 0.7085 | 1.138 |
| primary_population | reference | 10380 | 62.0 | 0.0063 | 0.0066 | 0.0063 | 0.0041 | 0.6719 | 1.1713 |
| primary_population | calendar | 10380 | 62.0 | 0.0063 | 0.0066 | 0.0063 | 0.0012 | 0.5956 | 0.9091 |

VARC-3 rule at the same landmarks (current echo against the reference study): all_landmarks: 4 positive, 9408 negative, 1010 indeterminate; primary_population: 0 positive, 9371 negative, 1009 indeterminate.
Landmarks stop before the first echo that meets the endpoint, so the rule is negative there by construction. It is compared as a surveillance policy in section 2 instead.

### Net benefit across the 5 to 15 percent band (primary population)

Net benefit per 1,000 landmarks (true positives net of weighted false positives). The ceiling is the 12-month SVD incidence, 6.3 per 1,000. The protocol band is 5% to 15%; thresholds below it are shown for context.

| threshold | model | calendar | reference | varc3_rule | assess_all |
|---|---|---|---|---|---|
| 0.01 | 1.37 | -0.37 | 0.82 | 0.0 | -3.7 |
| 0.02 | 0.66 | 0.0 | 0.0 | 0.0 | -13.95 |
| 0.03 | 0.42 | 0.0 | 0.0 | 0.0 | -24.4 |
| 0.04 | 0.06 | 0.0 | 0.0 | 0.0 | -35.07 |
| 0.05 | -0.09 | 0.0 | 0.0 | 0.0 | -45.96 |
| 0.06 | -0.03 | 0.0 | 0.0 | 0.0 | -57.09 |
| 0.07 | -0.02 | 0.0 | 0.0 | 0.0 | -68.46 |
| 0.08 | -0.02 | 0.0 | 0.0 | 0.0 | -80.07 |
| 0.09 | -0.01 | 0.0 | 0.0 | 0.0 | -91.94 |
| 0.1 | -0.01 | 0.0 | 0.0 | 0.0 | -104.07 |
| 0.11 | 0.0 | 0.0 | 0.0 | 0.0 | -116.48 |
| 0.12 | 0.0 | 0.0 | 0.0 | 0.0 | -129.16 |
| 0.13 | 0.0 | 0.0 | 0.0 | 0.0 | -142.14 |
| 0.14 | 0.0 | 0.0 | 0.0 | 0.0 | -155.42 |
| 0.15 | 0.0 | 0.0 | 0.0 | 0.0 | -169.02 |

## 2. The model as a scheduler (surveillance-policy simulation)

Each held-out patient's noise-free valve trajectory is replayed under a guideline schedule, and under the same schedule with KAIROS bringing the next echo forward (to 6 months) when the 12-month SVD risk is at or above the threshold. Guideline echoes are never removed. Detection is the first echo on which the VARC-3 rule is positive against the reference study.

| policy | patients | echoes_per_1000py | crossers | detected_pct | delay_median_months | delay_p90_months | early_positives_per_1000py | patients_alerted_pct |
|---|---|---|---|---|---|---|---|---|
| annual | 2390 | 887.9 | 266 | 82.3 | 6.1 | 11.6 | 2.0 | 0.0 |
| annual+kairos@0.02 | 2390 | 924.3 | 266 | 83.5 | 5.2 | 11.4 | 2.5 | 7.2 |
| annual+kairos@0.05 | 2390 | 890.1 | 266 | 82.3 | 6.1 | 11.6 | 2.1 | 0.8 |
| annual+kairos@0.1 | 2390 | 888.0 | 266 | 82.3 | 6.1 | 11.6 | 2.0 | 0.0 |
| acc_aha | 2390 | 437.5 | 266 | 53.4 | 12.0 | 37.2 | 0.9 | 0.0 |
| acc_aha+kairos@0.02 | 2390 | 472.6 | 266 | 58.6 | 8.6 | 36.0 | 1.8 | 5.0 |
| acc_aha+kairos@0.05 | 2390 | 438.9 | 266 | 54.5 | 11.5 | 36.9 | 1.0 | 0.3 |
| acc_aha+kairos@0.1 | 2390 | 437.6 | 266 | 53.4 | 12.0 | 37.2 | 0.9 | 0.0 |

### Lead time of KAIROS-guided surveillance over its base schedule (paired, per patient)

| policy | versus | crossers | detected_by_both | only_guided_detected | only_base_detected | lead_mean_months | lead_mean_ci95 | lead_median_months | earlier_pct | extra_echoes_per_1000py | extra_echoes_ci95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| annual+kairos@0.02 | annual | 266 | 216 | 6 | 3 | 0.64 | [0.34, 1.0] | 0.0 | 8.8 | 36.3 | [30.8, 43.04] |
| annual+kairos@0.05 | annual | 266 | 218 | 1 | 1 | 0.03 | [0.0, 0.09] | 0.0 | 0.5 | 2.2 | [1.21, 3.22] |
| annual+kairos@0.1 | annual | 266 | 219 | 0 | 0 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | 0.1 | [0.0, 0.3] |
| acc_aha+kairos@0.02 | acc_aha | 266 | 142 | 14 | 0 | 1.46 | [0.61, 2.73] | 0.0 | 7.0 | 35.1 | [27.38, 44.17] |
| acc_aha+kairos@0.05 | acc_aha | 266 | 142 | 3 | 0 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | 1.4 | [0.5, 2.4] |
| acc_aha+kairos@0.1 | acc_aha | 266 | 142 | 0 | 0 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | 0.1 | [0.0, 0.29] |

*Lead* is averaged over crossers both policies detect before follow-up ends; *only guided detected* counts crossers the base schedule missed altogether (death, replacement or the end of follow-up came first).

### By route

| policy | route | patients | echoes_per_1000py | crossers | detected_pct | delay_median_months | delay_p90_months |
|---|---|---|---|---|---|---|---|
| annual | SAVR | 1184 | 904.2 | 208 | 82.2 | 6.3 | 11.6 |
| annual | TAVR | 1206 | 866.3 | 58 | 82.8 | 4.6 | 17.4 |
| annual+kairos@0.02 | SAVR | 1184 | 967.2 | 208 | 83.7 | 5.4 | 11.1 |
| annual+kairos@0.02 | TAVR | 1206 | 867.4 | 58 | 82.8 | 4.6 | 17.4 |
| annual+kairos@0.05 | SAVR | 1184 | 908.0 | 208 | 82.2 | 6.3 | 11.5 |
| annual+kairos@0.05 | TAVR | 1206 | 866.3 | 58 | 82.8 | 4.6 | 17.4 |
| annual+kairos@0.1 | SAVR | 1184 | 904.4 | 208 | 82.2 | 6.3 | 11.6 |
| annual+kairos@0.1 | TAVR | 1206 | 866.3 | 58 | 82.8 | 4.6 | 17.4 |
| acc_aha | SAVR | 1184 | 128.0 | 208 | 45.2 | 18.8 | 41.7 |
| acc_aha | TAVR | 1206 | 866.3 | 58 | 82.8 | 4.6 | 17.4 |
| acc_aha+kairos@0.02 | SAVR | 1184 | 185.3 | 208 | 51.9 | 13.5 | 39.5 |
| acc_aha+kairos@0.02 | TAVR | 1206 | 867.4 | 58 | 82.8 | 4.6 | 17.4 |
| acc_aha+kairos@0.05 | SAVR | 1184 | 130.0 | 208 | 46.6 | 18.5 | 41.6 |
| acc_aha+kairos@0.05 | TAVR | 1206 | 866.3 | 58 | 82.8 | 4.6 | 17.4 |
| acc_aha+kairos@0.1 | SAVR | 1184 | 128.2 | 208 | 45.2 | 18.8 | 41.7 |
| acc_aha+kairos@0.1 | TAVR | 1206 | 866.3 | 58 | 82.8 | 4.6 | 17.4 |

## Reading this

- *Delay* is months from the latent (noise-free) threshold crossing to the first positive echo.
- *Lead* is how many months earlier the KAIROS-guided policy detects the same patient than its base schedule.
- *Early positives* are positive echoes before any latent crossing: measurement noise or thrombosis, not SVD.
- Assumptions: full attendance, a single positive study counts as detection, the regurgitation location is reported on 90 percent of echoes, echoes brought forward to 6 months. All labelled assumed.
- The simulated patients follow the generator's own model of deterioration, so the result shows what the model can do if that model of deterioration is right. It is not clinical evidence.
