# KAIROS revision: change log

Revision of the KAIROS proposal (original preserved as `kairos_proposal_original_v13.html`) into a submission for the Dyania Health Hackathon 2026. Applied 16 September 2026.

## Scope and positioning

- Repositioned KAIROS as a working research prototype with three explicit layers: real-record extraction (implemented), model development and evaluation on synthetic scenarios (demonstrated), and clinical validation in hospital cohorts (planned). Status tags (Implemented, Demonstrated, Planned, Unverified) applied throughout.
- Removed all claims of demonstrated clinical accuracy, outcome improvement or deployment readiness. Clinical validation is framed as the next phase, not as an unmet objective.
- Kept the central objective as building and demonstrating a durability prediction model; the extraction layer is a feasibility finding, not the project.
- Reduced emphasis on wearables, MIMIC cohort construction, 15- to 20-year individual prediction and multiple model families; all moved to an exploratory future-research appendix or dropped.

## Model

- One primary model: penalised cause-specific proportional hazards for SVD, death and non-SVD index-valve replacement, combined into cumulative incidence, with dynamic landmarking.
- Hierarchical device effects (frailty, partial pooling) relabelled as planned; the prototype uses design-class fixed effects with model and generation where event counts allow.
- Gradient boosting demoted to an optional comparison after the biomarker and anticoagulant modules; no other model family promised.
- Monotonic constraint on gradient change not imposed; constrained model kept as a sensitivity analysis.

## Timing and leakage

- Time zero redefined as the qualifying reference echocardiogram (30 to 180 days after implantation), resolving the contradiction between a fixed day-90 prediction and a reference window extending to day 180.
- Leakage rules stated: only information dated on or before the prediction time; patients meeting the endpoint leave the risk set; the echo establishing an endpoint is never a predictor of that event; patient-level grouping in resampling; preprocessing and model selection within training data; no model selection on the final external cohort.
- Year-only dates in the supplied records stated as unable to support landmark timing, exposure lags or longitudinal rates; day-level demonstrations use explicitly synthetic dates.

## Predicted quantities

- Principal output defined as the probability of SVD before death within the horizon, conditional on being alive with the index valve and SVD-free at prediction time; companion outputs defined as probability of death before SVD and probability of remaining alive with the index valve without SVD.
- Removed the earlier labelling of the complement of SVD cumulative incidence as the probability of being alive with a functioning valve.
- Supported horizons set to 1, 3 and 5 years with a 12-month near-term horizon for any earlier-assessment decision; 8 and 10 years exploratory; 15 to 20 years moved to future research.

## Endpoints and adjudication

- Primary endpoint retained: first adjudicated moderate or severe haemodynamic valve deterioration attributable to SVD by VARC-3, including stenotic, regurgitant and mixed phenotypes.
- Bioprosthetic valve failure corrected to the VARC-3 staging (stage 1 clinically expressive dysfunction or irreversible stage 3 HVD; stage 2 reintervention; stage 3 valve-related death) and split into all-cause and SVD-attributable BVF.
- Index-valve replacement for a non-SVD cause added as a competing event; medically treated, resolved thrombosis defined as an intercurrent state that returns the patient to the risk set rather than an absorbing event.
- Adjudication framework condensed: candidate identification, two reviewers with arbitration, uncertain class, mechanism-overlap rule, confirmation with pre-specified exceptions; a regurgitant endpoint is not invalidated by missing area or dimensionless index; published normal values are contextual and cannot replace the patient's own reference study for change-based staging.

## Biomarkers and anticoagulants

- Biomarker panel retained in full and organised into five modules (renal and metabolic; mineral metabolism; lipid-related susceptibility; cardiac response; inflammatory and molecular), each with rationale, bioprosthesis-specific versus extrapolated evidence, failure phenotype, availability in the supplied data, representation in the prototype and hypothetical status. No composite scores; core model runs without any module; absent markers are not imputed.
- Lp(a) presented as the example of conflicting, phenotype-specific evidence, distinguishing incident-SVD cohorts from reintervention-restricted studies.
- Anticoagulant module retained in detail: drug and class, indication, start, stop, interruption and switching, current and cumulative exposure, INR only where dated, timing relative to suspicious echo findings, prescription versus adherence. Blanket 90-day lag replaced by an explicit post-suspicion indicator and dual reporting, so recent thrombosis information is not erased. SVD and thrombosis-related dysfunction kept distinct; associations stated as predictive.
- Statement that factor Xa inhibitors are "neutral to favourable" replaced with a balanced account: reduced imaging-detected leaflet thrombosis (GALILEO-4D, ADAPT-TAVR non-significant, ATLANTIS-4D) against clinical harm without indication (GALILEO death HR 1.69) and increased bleeding (ENVISAGE-TAVI AF HR 1.40). No treatment recommendations.

## Simulation and evaluation

- Scenario set fixed at six: gradual stenotic, regurgitant or abrupt, high competing mortality, irregular surveillance, biomarker information meaningful, weak and absent, anticoagulant associations by mechanism and confounding including treatment after a suspicious echo.
- Each parameter labelled literature-informed, assumed or varied; published sub-distribution hazard ratios not used as cause-specific parameters without justification.
- Model ladder specified: reference (valve age and type), core, core plus biomarker modules, core plus anticoagulant history, core plus both, and core without serial echo where feasible; calibration and Brier prioritised, discrimination supporting, uncertainty and missing-data scenarios included.

## Demonstration and clinical framing

- Demonstration rebuilt around one longitudinal patient: record, passport, trajectory, updated risk, explanation and reliability statement; synthetic-trained predictions shown beside real extracted values labelled illustrative and unvalidated.
- Three messages separated: current abnormality requiring assessment, predicted near-term risk, overdue surveillance reminder. Guideline surveillance preserved; five-year risk no longer linked to appointment intervals.

## Factual corrections and consistency

- VARC-3 citation corrected to Généreux et al., Eur Heart J 2021;42:1825–1857 (co-published JACC 2021;77:2717–2746); the 2022 JACC paper previously cited as VARC-3 is the Pibarot et al. state-of-the-art review and is now cited as such.
- LOTUS and LOTUS Edge reclassified as mechanically expanded intra-annular TAVI in the document and, with the owner's authorisation on 16 September, at source in the device-table build script; the device table and both FDA summary tables were regenerated.
- Surveillance guidance stated by jurisdiction (ESC/EACTS yearly imaging; ACC/AHA 2020 echo at 5 and 10 years then yearly for surgical valves, yearly after TAVI); 2025 ESC/EACTS follow-up table marked unverified.
- Redo-surgery mortality attributed to its population and era (UK national registry, all reoperative aortic valve surgery 1996 to 2019: 4.8% elective, 11.8% urgent, 31.7% emergency, 50.7% salvage) and described as association, not preventable mortality.
- HVD, SVD, BVF and reintervention rates kept distinct wherever numbers are quoted; NOTION results qualified by population and device generation.
- Claim that all TAVI follow-up beyond ten years is extrapolation replaced with the statement that first-generation cohorts report outcomes to about twelve years in small surviving populations.
- Absolute novelty claim replaced with a qualified statement.
- Patient and note counts reconciled: 215 notes, 13 with deleted status (9 operative reports, 4 progress notes), 202 analysed; 117 patients; 97 with implant notes (53 surgical, 44 transcatheter), 15 history-only, 5 unspecified; 86 with explicit implant years; 110 with a named model; denominators stated per row. Years extending to 2027 explained as probable de-identification date shifting, marked inferred.
- Incomplete references completed where the source could be verified; remaining gaps listed in the unverified-details file rather than left as "verify later" notes in the text.

## Document

- Concise summary and README block moved to the top; operational timeline reduced to phased delivery without invented hours; revision history, review tables and agent references removed; print styles retained so tables do not clip.

## Addition 2026-09-17

- Biomarker modules: added dephosphorylated-uncarboxylated matrix Gla protein (dp-ucMGP) as an explicit vitamin K status marker, chosen over undercarboxylated osteocalcin, paired with cumulative vitamin K antagonist exposure in the anticoagulant module and simulated in the VKA scenario. Evidence labelled extrapolation (no bioprosthetic outcome study; AVADEC null for native valve calcification).

- 17 September: dp-ucMGP defined as a consumed model input (spline main effect plus VKA interaction inside the anticoagulant module, fitted on a full-cohort offset in the measured substudy, never imputed) with a dedicated synthetic check; previously it was only tracked.

## Implementation 2026-09-17: dp-ucMGP consumed by the model

- `src/kairos/modelling/vitamin_k.py`: dp-ucMGP (`dp_ucmgp`, pmol/L, assay recorded on `LabObservation.assay`) is read at each landmark, carried forward for at most 12 months and then marked stale; log value enters a restricted cubic spline main effect plus an interaction with cumulative VKA years, on the SVD hazard only. Never imputed: the step with the anticoagulant module is fitted on the full cohort, then an offset model (full-cohort SVD linear predictor as offset, ridge Cox, Breslow ties) estimates only the dp-ucMGP terms on measured rows. Centring keeps the substudy's expected SVD events unchanged. The test of the terms is a Wald test with a patient-bootstrap covariance, because stacked landmark rows are not independent (the nominal likelihood-ratio p-value is reported for contrast only).
- Predictions carry `dp_ucmgp` (`used`, `not_measured`, `stale`, `not_in_model`, with value, date and assay); the model card carries the substudy model; the demo shows both and the form accepts dp-ucMGP values.
- `anticoagulant_mechanism_confounding` now has two variants: `marker_mediated` (marker generated from VKA exposure and dialysis plus noise; SVD hazard depends on the latent marker, not the drug) and `marker_noise` (drug acts directly; marker is pure noise). The marker uses its own random stream, so other draws of every cohort are unchanged. Every scenario has a 30 % substudy (50 % in the anticoagulant scenario); the marker carries SVD information only in `marker_mediated`.
- `src/kairos/evaluation/vitamin_k.py`: patient-grouped cross-validated incremental Brier, IPA and calibration within measured rows, plus the mechanism check (VKA contrast fitted on measured rows without and with the dp-ucMGP terms; shrinkage reported, never tuned). Written by the evaluate job to `ladder.json` (`dp_ucmgp_substudy`) and `dp_ucmgp_substudy.csv`.
