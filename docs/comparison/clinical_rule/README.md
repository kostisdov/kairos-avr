# Comparison against the guideline threshold rule

The rule is the VARC-3 haemodynamic criteria applied to the current echocardiogram against the
patient's own reference study: a mean gradient rise of 10 mmHg or more to reach 20 mmHg or more,
together with the area or dimensionless-index criterion, or new or worsened intraprosthetic
regurgitation. It is what a cardiologist does today, and it is the comparator that decides whether
this model adds anything to practice.

**It has now been run.** Out-of-fold model predictions and comparator rows on the identical 10,701
landmarks from 2,390 patients, scenario `gradual_stenotic`.

| | Landmarks |
|---|---|
| Model produces a supported prediction | **10,701 of 10,701** |
| Rule returns an evaluable verdict | **0 of 10,701** |
| Rule positives | 0 |

The rule abstained everywhere. It requires a complete severity assessment before it will return a
verdict, and the synthetic generator does not populate every input it needs, above all the
confirmation that regurgitation is intraprosthetic rather than paravalvular. Faced with incomplete
inputs the comparator returns `indeterminate` rather than guessing, which is the behaviour it was
written to have.

## What this does and does not show

**It does show** that the two carry different data requirements. The model produced a supported
prediction on every landmark; the rule produced none. A threshold rule needs a complete study to
fire, and an incomplete study is the normal case in a real record.

**It does not show** that the rule is unusable in practice. This is a property of the synthetic
cohort, not evidence about patients. A real echocardiography report carries the confirmation fields
the generator omits.

**It does not yet give net benefit.** With zero rule positives the decision curve is degenerate, so
the net-benefit comparison that the protocol names as the primary clinical comparison remains
unmeasured. Closing it needs either a generator that populates the confirmation fields or a real
cohort, and it is the first analysis to run in the next phase.

Raw outputs are `report.md`, `coverage.json`, `paired_metrics.csv` and `decision_curves.csv`.
Produced by `scripts/evaluate_clinical_comparator.py`.
