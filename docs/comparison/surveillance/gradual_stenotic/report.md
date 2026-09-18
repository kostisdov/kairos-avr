# KAIROS against current practice: `gradual_stenotic`

Synthetic and illustrative. Out-of-fold throughout: no patient is scored or guided by a model trained on them.

## 1. The model as a risk score (12-month SVD, landmark level)

| population | score | n | n_events | observed | mean_predicted | brier | ipa | auc | cal_slope_logit |
|---|---|---|---|---|---|---|---|---|---|
| all_landmarks | model | 10701 | 33.0 | 0.0033 | 0.0034 | 0.0032 | 0.0205 | 0.9454 | 1.3387 |
| all_landmarks | reference | 10701 | 33.0 | 0.0033 | 0.0033 | 0.0033 | 0.0011 | 0.6541 | 0.9594 |
| all_landmarks | calendar | 10701 | 33.0 | 0.0033 | 0.0035 | 0.0033 | -0.0004 | 0.5449 | 0.4431 |
| primary_population | model | 10618 | 28.0 | 0.0028 | 0.0029 | 0.0027 | 0.0395 | 0.9414 | 1.6715 |
| primary_population | reference | 10618 | 28.0 | 0.0028 | 0.0033 | 0.0028 | 0.0015 | 0.7088 | 1.3222 |
| primary_population | calendar | 10618 | 28.0 | 0.0028 | 0.0035 | 0.0028 | 0.0001 | 0.612 | 0.8703 |

VARC-3 rule at the same landmarks (current echo against the reference study): all_landmarks: 5 positive, 9657 negative, 1039 indeterminate; primary_population: 0 positive, 9586 negative, 1032 indeterminate.
Landmarks stop before the first echo that meets the endpoint, so the rule is negative there by construction. It is compared as a surveillance policy in section 2 instead.

### Net benefit across the 5 to 15 percent band (primary population)

Net benefit per 1,000 landmarks (true positives net of weighted false positives). The ceiling is the 12-month SVD incidence, 2.8 per 1,000. The protocol band is 5% to 15%; thresholds below it are shown for context.

| threshold | model | calendar | reference | varc3_rule | assess_all |
|---|---|---|---|---|---|
| 0.01 | 1.27 | 0.0 | 0.0 | 0.0 | -7.27 |
| 0.02 | 0.79 | 0.0 | 0.0 | 0.0 | -17.55 |
| 0.03 | 0.56 | 0.0 | 0.0 | 0.0 | -28.04 |
| 0.04 | 0.28 | 0.0 | 0.0 | 0.0 | -38.75 |
| 0.05 | 0.11 | 0.0 | 0.0 | 0.0 | -49.69 |
| 0.06 | 0.14 | 0.0 | 0.0 | 0.0 | -60.85 |
| 0.07 | 0.05 | 0.0 | 0.0 | 0.0 | -72.26 |
| 0.08 | 0.05 | 0.0 | 0.0 | 0.0 | -83.91 |
| 0.09 | 0.07 | 0.0 | 0.0 | 0.0 | -95.82 |
| 0.1 | 0.07 | 0.0 | 0.0 | 0.0 | -108.0 |
| 0.11 | 0.07 | 0.0 | 0.0 | 0.0 | -120.45 |
| 0.12 | 0.07 | 0.0 | 0.0 | 0.0 | -133.18 |
| 0.13 | 0.08 | 0.0 | 0.0 | 0.0 | -146.21 |
| 0.14 | 0.1 | 0.0 | 0.0 | 0.0 | -159.53 |
| 0.15 | 0.1 | 0.0 | 0.0 | 0.0 | -173.17 |

## 2. The model as a scheduler (surveillance-policy simulation)

Each held-out patient's noise-free valve trajectory is replayed under a guideline schedule, and under the same schedule with KAIROS bringing the next echo forward (to 6 months) when the 12-month SVD risk is at or above the threshold. Guideline echoes are never removed. Detection is the first echo on which the VARC-3 rule is positive against the reference study.

| policy | patients | echoes_per_1000py | crossers | detected_pct | delay_median_months | delay_p90_months | early_positives_per_1000py | patients_alerted_pct |
|---|---|---|---|---|---|---|---|---|
| annual | 2390 | 886.3 | 182 | 61.5 | 7.1 | 14.6 | 4.5 | 0.0 |
| annual+kairos@0.02 | 2390 | 903.5 | 182 | 64.3 | 6.2 | 10.5 | 5.5 | 6.1 |
| annual+kairos@0.05 | 2390 | 890.2 | 182 | 62.1 | 6.7 | 12.3 | 4.9 | 1.7 |
| annual+kairos@0.1 | 2390 | 887.0 | 182 | 61.5 | 6.9 | 14.5 | 4.7 | 0.3 |
| acc_aha | 2390 | 434.3 | 182 | 34.1 | 9.9 | 21.8 | 1.9 | 0.0 |
| acc_aha+kairos@0.02 | 2390 | 442.5 | 182 | 44.5 | 8.9 | 20.6 | 2.5 | 1.8 |
| acc_aha+kairos@0.05 | 2390 | 436.5 | 182 | 37.9 | 9.2 | 21.0 | 1.9 | 0.4 |
| acc_aha+kairos@0.1 | 2390 | 434.3 | 182 | 34.1 | 9.9 | 21.8 | 1.9 | 0.0 |

### Lead time of KAIROS-guided surveillance over its base schedule (paired, per patient)

| policy | versus | crossers | detected_by_both | only_guided_detected | only_base_detected | lead_mean_months | lead_mean_ci95 | lead_median_months | earlier_pct | extra_echoes_per_1000py | extra_echoes_ci95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| annual+kairos@0.02 | annual | 182 | 105 | 12 | 7 | 1.83 | [1.16, 2.64] | 0.0 | 24.8 | 17.2 | [13.86, 20.61] |
| annual+kairos@0.05 | annual | 182 | 110 | 3 | 2 | 0.76 | [0.38, 1.22] | 0.0 | 10.9 | 3.9 | [2.49, 5.22] |
| annual+kairos@0.1 | annual | 182 | 111 | 1 | 1 | 0.22 | [0.0, 0.61] | 0.0 | 1.8 | 0.7 | [0.22, 1.29] |
| acc_aha+kairos@0.02 | acc_aha | 182 | 61 | 20 | 1 | 0.1 | [0.0, 0.33] | 0.0 | 1.6 | 8.2 | [5.61, 11.32] |
| acc_aha+kairos@0.05 | acc_aha | 182 | 62 | 7 | 0 | 0.1 | [0.0, 0.3] | 0.0 | 1.6 | 2.2 | [0.92, 3.97] |
| acc_aha+kairos@0.1 | acc_aha | 182 | 62 | 0 | 0 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 | 0.0 | [0.0, 0.0] |

*Lead* is averaged over crossers both policies detect before follow-up ends; *only guided detected* counts crossers the base schedule missed altogether (death, replacement or the end of follow-up came first).

### By route

| policy | route | patients | echoes_per_1000py | crossers | detected_pct | delay_median_months | delay_p90_months |
|---|---|---|---|---|---|---|---|
| annual | SAVR | 1184 | 901.9 | 148 | 63.5 | 7.0 | 14.4 |
| annual | TAVR | 1206 | 865.0 | 34 | 52.9 | 7.4 | 18.2 |
| annual+kairos@0.02 | SAVR | 1184 | 930.2 | 148 | 67.6 | 6.1 | 10.2 |
| annual+kairos@0.02 | TAVR | 1206 | 867.2 | 34 | 50.0 | 6.9 | 15.5 |
| annual+kairos@0.05 | SAVR | 1184 | 908.6 | 148 | 64.2 | 6.6 | 12.1 |
| annual+kairos@0.05 | TAVR | 1206 | 865.1 | 34 | 52.9 | 7.4 | 14.6 |
| annual+kairos@0.1 | SAVR | 1184 | 903.1 | 148 | 63.5 | 6.9 | 14.2 |
| annual+kairos@0.1 | TAVR | 1206 | 865.0 | 34 | 52.9 | 7.4 | 18.2 |
| acc_aha | SAVR | 1184 | 125.7 | 148 | 29.7 | 10.9 | 23.3 |
| acc_aha | TAVR | 1206 | 865.0 | 34 | 52.9 | 7.4 | 18.2 |
| acc_aha+kairos@0.02 | SAVR | 1184 | 136.4 | 148 | 43.2 | 10.2 | 20.5 |
| acc_aha+kairos@0.02 | TAVR | 1206 | 867.2 | 34 | 50.0 | 6.9 | 15.5 |
| acc_aha+kairos@0.05 | SAVR | 1184 | 128.8 | 148 | 34.5 | 10.5 | 21.9 |
| acc_aha+kairos@0.05 | TAVR | 1206 | 865.1 | 34 | 52.9 | 7.4 | 14.6 |
| acc_aha+kairos@0.1 | SAVR | 1184 | 125.7 | 148 | 29.7 | 10.9 | 23.3 |
| acc_aha+kairos@0.1 | TAVR | 1206 | 865.0 | 34 | 52.9 | 7.4 | 18.2 |

## Reading this

- *Delay* is months from the latent (noise-free) threshold crossing to the first positive echo.
- *Lead* is how many months earlier the KAIROS-guided policy detects the same patient than its base schedule.
- *Early positives* are positive echoes before any latent crossing: measurement noise or thrombosis, not SVD.
- Assumptions: full attendance, a single positive study counts as detection, the regurgitation location is reported on 90 percent of echoes, echoes brought forward to 6 months. All labelled assumed.
- The simulated patients follow the generator's own model of deterioration, so the result shows what the model can do if that model of deterioration is right. It is not clinical evidence.
