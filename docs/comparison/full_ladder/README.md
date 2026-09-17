# Family comparison across the full ladder

A later run extended the penalised Cox against gradient boosting comparison from three ladder steps
to **eleven**, covering every module block. The conclusion is unchanged: **gradient boosting was
promoted at no step.**

| Ladder step | Mean Brier difference | 95% interval |
|---|---|---|
| reference | -6.90e-05 | [-1.47e-04, +2.43e-05] |
| core | +3.51e-04 | [-7.47e-04, +1.32e-03] |
| core_plus_renal_metabolic | +1.32e-04 | [-9.09e-04, +1.20e-03] |
| core_plus_mineral | +3.29e-04 | [-7.21e-04, +1.23e-03] |
| core_plus_lipid | +3.51e-04 | [-7.67e-04, +1.22e-03] |
| core_plus_cardiac | +4.10e-04 | [-6.68e-04, +1.38e-03] |
| core_plus_inflammatory | +3.45e-04 | [-7.18e-04, +1.30e-03] |
| core_plus_anticoagulant | +2.81e-05 | [-1.85e-03, +7.96e-04] |
| core_plus_biomarkers | +2.39e-04 | [-6.76e-04, +1.21e-03] |
| core_plus_both | +9.68e-05 | [-1.36e-03, +1.10e-03] |
| core_without_serial_echo | **-2.97e-04** | **[-6.27e-04, -9.78e-05]** |

A positive difference means boosting scored worse. Ten of the eleven intervals span zero, so the two
families are indistinguishable at those steps.

**One exception worth stating.** At `core_without_serial_echo` the interval lies entirely below zero,
so boosting genuinely improved there. That is the one configuration with no serial echocardiography,
which is precisely the configuration we do not propose: it is the ablation that shows what updating
contributes. Boosting compensating slightly when the echo trajectory is removed does not argue for
boosting in the model we actually put forward.

These records carry the decision label `not_eligible`, meaning the run did not meet the gate for a
binding promotion decision, rather than the `reference`, `core` and `core_plus_both` records in the
parent directory, which reached `retain_cox`. The headline figures quoted in `model/approach.md` and
on the slides come from those three, because they carry an actual decision. This directory is the
broader supporting evidence.
