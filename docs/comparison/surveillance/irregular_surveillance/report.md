# KAIROS against current practice: `irregular_surveillance`

Synthetic and illustrative. Out-of-fold throughout: no patient is scored or guided by a model trained on them.

## 1. The model as a risk score (12-month SVD, landmark level)

| population | score | n | n_events | observed | mean_predicted | brier | ipa | auc | cal_slope_logit |
|---|---|---|---|---|---|---|---|---|---|
| all_landmarks | model | 8239 | 29.0 | 0.0037 | 0.0038 | 0.0037 | 0.0131 | 0.8696 | 1.323 |
| all_landmarks | reference | 8239 | 29.0 | 0.0037 | 0.0037 | 0.0037 | 0.0021 | 0.6602 | 1.0737 |
| all_landmarks | calendar | 8239 | 29.0 | 0.0037 | 0.0038 | 0.0037 | -0.001 | 0.4571 | 0.1032 |
| primary_population | model | 8181 | 25.0 | 0.0032 | 0.0034 | 0.0032 | 0.0157 | 0.8542 | 1.5966 |
| primary_population | reference | 8181 | 25.0 | 0.0032 | 0.0037 | 0.0032 | 0.0025 | 0.7066 | 1.3047 |
| primary_population | calendar | 8181 | 25.0 | 0.0032 | 0.0038 | 0.0032 | -0.0007 | 0.4912 | 0.3654 |

VARC-3 rule at the same landmarks (current echo against the reference study): all_landmarks: 5 positive, 7421 negative, 813 indeterminate; primary_population: 0 positive, 7371 negative, 810 indeterminate.
Landmarks stop before the first echo that meets the endpoint, so the rule is negative there by construction. It is compared as a surveillance policy in section 2 instead.

### Net benefit across the 5 to 15 percent band (primary population)

Net benefit per 1,000 landmarks (true positives net of weighted false positives). The ceiling is the 12-month SVD incidence, 3.2 per 1,000. The protocol band is 5% to 15%; thresholds below it are shown for context.

| threshold | model | calendar | reference | varc3_rule | assess_all |
|---|---|---|---|---|---|
| 0.01 | 1.23 | 0.0 | -0.09 | 0.0 | -6.83 |
| 0.02 | 0.6 | 0.0 | 0.0 | 0.0 | -17.11 |
| 0.03 | 0.15 | 0.0 | 0.0 | 0.0 | -27.59 |
| 0.04 | 0.17 | 0.0 | 0.0 | 0.0 | -38.3 |
| 0.05 | -0.05 | 0.0 | 0.0 | 0.0 | -49.23 |
| 0.06 | -0.02 | 0.0 | 0.0 | 0.0 | -60.39 |
| 0.07 | -0.01 | 0.0 | 0.0 | 0.0 | -71.79 |
| 0.08 | -0.01 | 0.0 | 0.0 | 0.0 | -83.44 |
| 0.09 | -0.01 | 0.0 | 0.0 | 0.0 | -95.34 |
| 0.1 | -0.01 | 0.0 | 0.0 | 0.0 | -107.51 |
| 0.11 | 0.0 | 0.0 | 0.0 | 0.0 | -119.96 |
| 0.12 | 0.0 | 0.0 | 0.0 | 0.0 | -132.68 |
| 0.13 | 0.0 | 0.0 | 0.0 | 0.0 | -145.7 |
| 0.14 | 0.0 | 0.0 | 0.0 | 0.0 | -159.03 |
| 0.15 | 0.0 | 0.0 | 0.0 | 0.0 | -172.66 |

## 2. The model as a scheduler (surveillance-policy simulation)

Each held-out patient's noise-free valve trajectory is replayed under a guideline schedule, and under the same schedule with KAIROS bringing the next echo forward (to 6 months) when the 12-month SVD risk is at or above the threshold. Guideline echoes are never removed. Detection is the first echo on which the VARC-3 rule is positive against the reference study.

| policy | patients | echoes_per_1000py | crossers | detected_pct | delay_median_months | delay_p90_months | early_positives_per_1000py | patients_alerted_pct |
|---|---|---|---|---|---|---|---|---|
| annual | 2390 | 887.0 | 214 | 71.0 | 6.6 | 12.0 | 3.8 | 0.0 |
| annual+kairos@0.02 | 2390 | 901.4 | 214 | 73.4 | 5.9 | 11.3 | 4.6 | 4.9 |
| annual+kairos@0.05 | 2390 | 889.5 | 214 | 72.4 | 6.4 | 11.6 | 4.0 | 1.0 |
| annual+kairos@0.1 | 2390 | 887.4 | 214 | 71.5 | 6.7 | 11.9 | 3.8 | 0.3 |
| acc_aha | 2390 | 435.4 | 214 | 41.1 | 13.4 | 33.9 | 1.5 | 0.0 |
| acc_aha+kairos@0.02 | 2390 | 443.8 | 214 | 47.7 | 10.3 | 33.0 | 2.3 | 1.8 |
| acc_aha+kairos@0.05 | 2390 | 436.7 | 214 | 43.5 | 12.0 | 33.2 | 1.6 | 0.3 |
| acc_aha+kairos@0.1 | 2390 | 435.9 | 214 | 42.1 | 13.4 | 33.4 | 1.5 | 0.1 |

### Lead time of KAIROS-guided surveillance over its base schedule (paired, per patient)

| policy | versus | crossers | detected_by_both | only_guided_detected | only_base_detected | lead_mean_months | lead_mean_ci95 | lead_median_months | earlier_pct | extra_echoes_per_1000py | extra_echoes_ci95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| annual+kairos@0.02 | annual | 214 | 146 | 11 | 6 | 0.74 | [0.38, 1.12] | 0.0 | 11.0 | 14.4 | [11.41, 17.12] |
| annual+kairos@0.05 | annual | 214 | 151 | 4 | 1 | 0.16 | [0.0, 0.31] | 0.0 | 2.6 | 2.5 | [1.48, 3.7] |
| annual+kairos@0.1 | annual | 214 | 152 | 1 | 0 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | 0.4 | [0.09, 0.87] |
| acc_aha+kairos@0.02 | acc_aha | 214 | 87 | 15 | 1 | 0.07 | [0.0, 0.24] | 0.0 | 1.1 | 8.4 | [5.55, 11.79] |
| acc_aha+kairos@0.05 | acc_aha | 214 | 88 | 5 | 0 | 0.07 | [0.0, 0.21] | 0.0 | 1.1 | 1.3 | [0.34, 2.55] |
| acc_aha+kairos@0.1 | acc_aha | 214 | 88 | 2 | 0 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | 0.5 | [0.0, 1.16] |

*Lead* is averaged over crossers both policies detect before follow-up ends; *only guided detected* counts crossers the base schedule missed altogether (death, replacement or the end of follow-up came first).

### By route

| policy | route | patients | echoes_per_1000py | crossers | detected_pct | delay_median_months | delay_p90_months |
|---|---|---|---|---|---|---|---|
| annual | SAVR | 1184 | 902.9 | 174 | 72.4 | 6.7 | 11.7 |
| annual | TAVR | 1206 | 865.6 | 40 | 65.0 | 5.0 | 17.6 |
| annual+kairos@0.02 | SAVR | 1184 | 926.7 | 174 | 75.9 | 6.1 | 10.9 |
| annual+kairos@0.02 | TAVR | 1206 | 867.3 | 40 | 62.5 | 3.9 | 15.7 |
| annual+kairos@0.05 | SAVR | 1184 | 907.2 | 174 | 74.1 | 6.6 | 11.5 |
| annual+kairos@0.05 | TAVR | 1206 | 865.6 | 40 | 65.0 | 5.0 | 15.1 |
| annual+kairos@0.1 | SAVR | 1184 | 903.6 | 174 | 73.0 | 6.7 | 11.6 |
| annual+kairos@0.1 | TAVR | 1206 | 865.6 | 40 | 65.0 | 5.0 | 17.6 |
| acc_aha | SAVR | 1184 | 126.5 | 174 | 35.6 | 16.6 | 38.3 |
| acc_aha | TAVR | 1206 | 865.6 | 40 | 65.0 | 5.0 | 17.6 |
| acc_aha+kairos@0.02 | SAVR | 1184 | 138.0 | 174 | 44.3 | 14.1 | 36.0 |
| acc_aha+kairos@0.02 | TAVR | 1206 | 867.3 | 40 | 62.5 | 3.9 | 15.7 |
| acc_aha+kairos@0.05 | SAVR | 1184 | 128.1 | 174 | 38.5 | 15.1 | 37.3 |
| acc_aha+kairos@0.05 | TAVR | 1206 | 865.6 | 40 | 65.0 | 5.0 | 15.1 |
| acc_aha+kairos@0.1 | SAVR | 1184 | 127.0 | 174 | 36.8 | 15.6 | 37.9 |
| acc_aha+kairos@0.1 | TAVR | 1206 | 865.6 | 40 | 65.0 | 5.0 | 17.6 |

## Reading this

- *Delay* is months from the latent (noise-free) threshold crossing to the first positive echo.
- *Lead* is how many months earlier the KAIROS-guided policy detects the same patient than its base schedule.
- *Early positives* are positive echoes before any latent crossing: measurement noise or thrombosis, not SVD.
- Assumptions: full attendance, a single positive study counts as detection, the regurgitation location is reported on 90 percent of echoes, echoes brought forward to 6 months. All labelled assumed.
- The simulated patients follow the generator's own model of deterioration, so the result shows what the model can do if that model of deterioration is right. It is not clinical evidence.
