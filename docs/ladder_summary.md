### IPCW Brier at 5 years (lower is better), synthetic scenarios

| scenario | reference | core | core_plus_biomarkers | core_plus_anticoagulant | core_plus_both | core_without_serial_echo | n events (5 y) |
|---|---|---|---|---|---|---|---|
| anticoagulant_mechanism_confounding | 0.0621 | 0.0608 | 0.0625 | 0.0607 | 0.0623 | 0.0612 | 198 |
| biomarker_information/absent | 0.0490 | 0.0478 | 0.0477 | 0.0475 | 0.0475 | 0.0501 | 159 |
| biomarker_information/meaningful | 0.0524 | 0.0476 | 0.0480 | 0.0468 | 0.0472 | 0.0543 | 178 |
| biomarker_information/unmeasured | 0.0642 | 0.0610 | 0.0630 | 0.0609 | 0.0630 | 0.0641 | 227 |
| biomarker_information/weak | 0.0504 | 0.0485 | 0.0495 | 0.0487 | 0.0497 | 0.0516 | 168 |
| gradual_stenotic | 0.0559 | 0.0472 | 0.0489 | 0.0469 | 0.0485 | 0.0575 | 198 |
| high_competing_mortality | 0.0041 | 0.0042 | 0.0040 | 0.0042 | 0.0039 | 0.0046 | 3 |
| irregular_surveillance | 0.0639 | 0.0602 | 0.0590 | 0.0596 | 0.0584 | 0.0621 | 189 |
| regurgitant_abrupt | 0.0776 | 0.0782 | 0.0783 | 0.0785 | 0.0790 | 0.0792 | 267 |

### Calibration and discrimination of core_plus_both at 5 years

| scenario | observed | mean predicted | cal-in-large | cal slope | AUC (95% CI) | IPA |
|---|---|---|---|---|---|---|
| gradual_stenotic | 0.066 | 0.084 | -0.018 | 0.88 | 0.80 (0.77-0.83) | 0.212 |
| regurgitant_abrupt | 0.093 | 0.108 | -0.015 | 0.67 | 0.74 (0.66-0.80) | 0.067 |
| high_competing_mortality | 0.003 | 0.011 | -0.008 | -0.04 | 0.00 (0.00-0.00) | -0.316 |
| irregular_surveillance | 0.079 | 0.108 | -0.029 | 0.93 | 0.79 (0.73-0.83) | 0.194 |
| biomarker_information/meaningful | 0.062 | 0.069 | -0.007 | 0.89 | 0.79 (0.72-0.87) | 0.187 |
| biomarker_information/weak | 0.056 | 0.072 | -0.016 | 0.68 | 0.73 (0.67-0.80) | 0.058 |
| biomarker_information/absent | 0.054 | 0.072 | -0.018 | 0.76 | 0.67 (0.58-0.75) | 0.068 |
| biomarker_information/unmeasured | 0.078 | 0.093 | -0.016 | 0.78 | 0.75 (0.67-0.80) | 0.120 |
| anticoagulant_mechanism_confounding | 0.070 | 0.094 | -0.025 | 0.64 | 0.66 (0.57-0.75) | 0.037 |

### Module availability and anticoagulant coefficients

- **anticoagulant_mechanism_confounding**: modules off: none; SVD anticoagulant coefficients: {'ac_cum_vka_years': 0.07, 'ac_post_suspicion': -0.09, 'ac_class_current__DAPT': -0.16, 'ac_class_current__FXa': -0.18, 'ac_class_current__VKA': 0.19, 'ac_indication__suspected_valve_thrombosis': -0.98, 'ac_current_status__past': 0.82}; thrombosis outcome: {'ac_cum_oac_years': -0.06, 'ac_post_suspicion': -0.37, 'on_antiplatelet': 0.11, 'ac_class_current__DAPT': -0.14, 'ac_class_current__SAPT': 0.13, 'ac_class_current__VKA': -0.13, 'ac_indication__AF': -0.1, 'ac_indication__none': 0.11, 'ac_indication__suspected_valve_thrombosis': -0.31, 'ac_current_status__current': -0.11, 'ac_current_status__never': 0.12, 'ac_current_status__past': -0.46}
- **biomarker_information/absent**: modules off: none; SVD anticoagulant coefficients: {'ac_days_since_change': 0.06, 'ac_post_suspicion': -0.44, 'ac_class_current__DAPT': -0.09, 'ac_class_current__FXa': 0.07, 'ac_indication__other': -1.37, 'ac_current_status__past': -0.28}; thrombosis outcome: {'ac_post_suspicion': -0.19, 'on_antiplatelet': 0.06, 'ac_class_current__DAPT': 0.07, 'ac_class_current__VKA': -0.12, 'ac_indication__AF': -0.06, 'ac_indication__none': 0.06, 'ac_indication__other': -0.25, 'ac_current_status__current': -0.06, 'ac_current_status__never': 0.07, 'ac_current_status__past': -0.19}
- **biomarker_information/meaningful**: modules off: none; SVD anticoagulant coefficients: {'ac_post_suspicion': -0.37, 'ac_class_current__FXa': 0.13, 'ac_class_current__VKA': -0.2, 'ac_indication__none': 0.06, 'ac_indication__other': -1.21, 'ac_current_status__current': -0.06, 'ac_current_status__never': 0.07, 'ac_current_status__past': -0.17}; thrombosis outcome: {'ac_post_suspicion': -0.18, 'on_antiplatelet': 0.08, 'ac_class_current__DAPT': 0.25, 'ac_class_current__FXa': -0.12, 'ac_indication__AF': -0.08, 'ac_indication__none': 0.09, 'ac_indication__other': -0.17, 'ac_current_status__current': -0.09, 'ac_current_status__never': 0.09, 'ac_current_status__past': -0.19}
- **biomarker_information/unmeasured**: modules off: ['biomarker_mineral', 'biomarker_lipid']; SVD anticoagulant coefficients: {'ac_class_current__DAPT': -0.08, 'ac_class_current__FXa': -0.27, 'ac_class_current__VKA': 0.17, 'ac_indication__other': -0.79, 'ac_current_status__other': 0.56}; thrombosis outcome: {'ac_post_suspicion': -0.17, 'on_antiplatelet': 0.07, 'ac_class_current__DAPT': 0.11, 'ac_class_current__VKA': -0.08, 'ac_indication__AF': -0.07, 'ac_indication__none': 0.07, 'ac_indication__other': -0.21, 'ac_current_status__current': -0.07, 'ac_current_status__never': 0.07, 'ac_current_status__other': -0.16}
- **biomarker_information/weak**: modules off: none; SVD anticoagulant coefficients: {'ac_post_suspicion': -0.47, 'ac_class_current__DAPT': -0.14, 'ac_indication__other': -1.79, 'ac_current_status__past': -0.28}; thrombosis outcome: {'ac_post_suspicion': -0.11, 'ac_class_current__DAPT': 0.05, 'ac_class_current__VKA': -0.06, 'ac_indication__other': -0.18, 'ac_current_status__past': -0.1}
- **gradual_stenotic**: modules off: none; SVD anticoagulant coefficients: {'ac_cum_vka_years': 0.05, 'ac_post_suspicion': -0.11, 'ac_class_current__FXa': -0.09, 'ac_class_current__SAPT': 0.05, 'ac_indication__none': 0.06, 'ac_indication__other': -2.5, 'ac_current_status__current': -0.06, 'ac_current_status__past': 0.65}; thrombosis outcome: {'ac_post_suspicion': -0.22, 'on_antiplatelet': 0.1, 'ac_class_current__DAPT': 0.15, 'ac_class_current__FXa': -0.13, 'ac_class_current__SAPT': 0.05, 'ac_indication__AF': -0.1, 'ac_indication__none': 0.1, 'ac_indication__other': -0.17, 'ac_current_status__current': -0.1, 'ac_current_status__never': 0.11, 'ac_current_status__past': -0.25}
- **high_competing_mortality**: modules off: none; SVD anticoagulant coefficients: {'ac_cum_vka_years': 0.12, 'ac_cum_oac_years': 0.06, 'ac_post_suspicion': -0.2, 'ac_class_current__FXa': -0.14, 'ac_class_current__VKA': 0.17, 'ac_indication__other': -0.37, 'ac_current_status__other': -0.1}; thrombosis outcome: {'ac_post_suspicion': -0.25, 'ac_class_current__DAPT': 0.1, 'ac_class_current__FXa': -0.05, 'ac_class_current__SAPT': -0.06, 'ac_indication__other': -0.28, 'ac_current_status__other': -0.23}
- **irregular_surveillance**: modules off: none; SVD anticoagulant coefficients: {'ac_days_since_change': 0.08, 'ac_post_suspicion': -0.12, 'ac_class_current__DAPT': -0.08, 'ac_class_current__VKA': 0.09, 'ac_indication__AF': 0.06, 'ac_indication__other': -2.25, 'ac_current_status__never': -0.06, 'ac_current_status__past': 0.81}; thrombosis outcome: {'ac_post_suspicion': -0.16, 'ac_class_current__DAPT': 0.12, 'ac_class_current__FXa': 0.07, 'ac_class_current__VKA': -0.06, 'ac_indication__other': -0.18, 'ac_current_status__past': -0.16}
- **regurgitant_abrupt**: modules off: none; SVD anticoagulant coefficients: {'ac_cum_vka_years': 0.08, 'ac_cum_oac_years': 0.07, 'ac_post_suspicion': 0.95, 'ac_class_current__FXa': -0.07, 'ac_class_current__VKA': 0.08, 'ac_indication__other': -0.4, 'ac_current_status__past': 0.14}; thrombosis outcome: {'ac_post_suspicion': -0.23, 'ac_class_current__DAPT': 0.06, 'ac_class_current__FXa': -0.08, 'ac_indication__other': -0.25, 'ac_current_status__never': 0.06, 'ac_current_status__past': -0.24}
