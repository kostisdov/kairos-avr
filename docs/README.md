# Documentation

A reading order for the KAIROS documents. The deliverables themselves are linked from the
repository [README](../README.md): the study protocol, the model approach, the data plan and the
proof-of-concept notebook.

## Start here

| Document | What it answers |
|---|---|
| [`kairos_proposal_revised.html`](kairos_proposal_revised.html) | The scientific specification: endpoint, time zero, the predicted quantities, leakage rules, module structure |
| [`comparison/surveillance/README.md`](comparison/surveillance/README.md) | How KAIROS compares with current practice: lead time and echo burden under the ESC/EACTS and ACC/AHA schedules |
| [`ladder_summary.md`](ladder_summary.md) | Brier, calibration and discrimination of every model-ladder step on every synthetic scenario |
| [`comparison/validation_splits/README.md`](comparison/validation_splits/README.md) | Temporal and leave-one-design-class-out validation, and the surveillance-blinded and R4 comparators |
| [`deviations.md`](deviations.md) | Every departure from the specification and the protocol, with the reason and its consequence |
| [`kairos_unverified.md`](kairos_unverified.md) | Statements taken from secondary sources that we have not verified |

## Results

- [`comparison/README.md`](comparison/README.md): penalised Cox against gradient boosting, the pre-specified challenger.
- [`comparison/full_ladder/`](comparison/full_ladder/): the family decision at every ladder step.
- [`comparison/clinical_rule/`](comparison/clinical_rule/): the first rule comparison, superseded by `comparison/surveillance/`, kept because it shows why.
- [`model_cards/`](model_cards/): one card per model family.

## Designs, in the order they were written

1. [`kairos_change_log.md`](kairos_change_log.md): how the original proposal became the submission.
2. [`kairos_azure_build_design.md`](kairos_azure_build_design.md): repository layout, services, Azure resources and milestones (M0 to M5).
3. [`kairos_model_change_design.md`](kairos_model_change_design.md): the model change requests, numbered **CR-01** onwards.
4. [`kairos_model_change_detailed_design.md`](kairos_model_change_detailed_design.md): the work packages that implement them, numbered **WP-A1** onwards.
5. [`kairos_calibrated_generator_design.md`](kairos_calibrated_generator_design.md): the evidence-calibrated synthetic generator.
6. [`kairos_data_standardization_service_design.md`](kairos_data_standardization_service_design.md): the proposed clinical data standardisation service.
7. [`kairos_clinical_comparator_design.md`](kairos_clinical_comparator_design.md), [`kairos_varc3_comparator_implementation_design.md`](kairos_varc3_comparator_implementation_design.md) and its [report](kairos_varc3_comparator_implementation_report.md): the standalone VARC-3 comparator and the patient summary.

Code comments cite these identifiers (CR-xx, WP-xx, Mx, "design rule 0.x"); the documents above
define them.

## Data and operations

- [`data_build_report.md`](data_build_report.md) and [`data_build_log.md`](data_build_log.md): how the public reference tables and the aggregate extracts were built.
- [`extraction_pilot_audit_instructions.md`](extraction_pilot_audit_instructions.md) and [`km_digitisation_instructions.md`](km_digitisation_instructions.md): instructions for the manual steps.
- [`runbook.md`](runbook.md): running KAIROS locally and on Azure.
- [`schemas/`](schemas/): JSON schemas of every service payload, checked for drift in CI.
- [`archive/`](archive/): the original proposal, superseded by the revised one.
