# The protocol's comparators, and validation beyond random splits

Produced by `scripts/evaluate_validation_splits.py` (design in
`src/kairos/evaluation/validation_splits.py`, comparators in `config/model.yaml: protocol_comparators`).
Gradual stenotic scenario, SVD at 5 years, pooled landmark rows, synthetic and illustrative. Full
tables in [`gradual_stenotic/report.md`](gradual_stenotic/report.md).

## Against the protocol's comparator ladder

The protocol (section 5) makes R4, the current mean gradient and its change from the reference
study refreshed at every landmark, the comparator that carries the incremental-value claim.

| Comparator | Brier | IPA | AUC |
|---|---|---|---|
| R1 valve age and type | 0.0386 | 0.024 | 0.69 |
| R2 pre-implant (patient and prosthesis, no echo) | 0.0373 | 0.056 | 0.68 |
| R3 reference echo, fitted once | 0.0374 | 0.053 | 0.68 |
| R4 current gradient and its change | 0.0372 | 0.060 | 0.71 |
| **KAIROS core** | **0.0354** | **0.105** | **0.73** |

KAIROS against R4: Brier difference **-0.0018 (95% CI -0.0030 to -0.0007)**, patient bootstrap.
KAIROS beats the single-marker comparator, so the incremental-value claim holds on this scenario.
R3 does not beat R4 (+0.0003, CI -0.0008 to +0.0014): a reference echo read once adds nothing over
following the gradient.

## Is the advantage biological, or surveillance triage?

The protocol's test: remove the surveillance terms and see whether the advantage over R4 survives.

- **Protocol variant** (time since the last echo and the overdue flag removed): identical to KAIROS.
  Landmarks sit on echo dates, so both terms are constant at every landmark and the model already
  drops them. The protocol's variant cannot answer its own question in this design.
- **Stricter variant**, also without the number of echoes so far and the interval since the previous
  one: Brier 0.0354, the same as KAIROS, and the same -0.0018 against R4.

Nothing in KAIROS's advantage comes from how often a patient is imaged. On synthetic data this is
partly by construction, because the generator's visit schedule does not depend on deterioration in
this scenario; `irregular_surveillance`, where it does, is the harder test.

## Temporal split

Fitted on 1,284 patients implanted 2012 to 2016, scored on 1,106 implanted 2017 to 2020.

| Comparator | Brier | IPA | AUC | Calibration slope |
|---|---|---|---|---|
| R1 valve age and type | 0.0380 | 0.031 | 0.74 | 1.33 |
| R4 current gradient and its change | 0.0366 | 0.068 | 0.73 | 1.16 |
| **KAIROS core** | **0.0337** | **0.142** | **0.78** | 1.18 |

KAIROS holds up on later implants. It over-predicts (mean 5.5 percent against 4.1 observed), the same
direction as in cross-validation.

## Leave one design class out

Each design class with at least 10 SVD event patients is held out and scored by a model that never
saw it. Two classes qualify.

| Held-out class | Model | Brier | IPA | Calibration slope |
|---|---|---|---|---|
| Externally mounted pericardial | R1 valve age and type | 0.1037 | -0.039 | 2.19 |
| | **KAIROS core** | **0.0921** | **0.076** | 1.03 |
| Stented bovine pericardial, internally mounted | R1 valve age and type | 0.0383 | -0.051 | 2.27 |
| | **KAIROS core** | **0.0360** | **0.011** | 1.15 |

Without its own valve class, the valve-age-and-type model is worse than the class's average risk
(negative IPA). KAIROS stays useful because the serial echo carries what the class label no longer can.
The externally mounted class, which fails early, is under-predicted by both (7.0 percent predicted
against 11.2 observed), as a model that never saw an early-failing class should be. The other seven
classes have too few events to score.

## Not claimed

Synthetic scenarios, one seed. The comparators are fitted with the same penalised Cox machinery as
KAIROS, so the comparison is of information, not of method. R5, the guideline rule, is compared as a
surveillance policy in [`../surveillance/`](../surveillance/).
