# Study Protocol

> **Instructions:** This is the main deliverable. Replace every `> *Fill in:*` block with your team's content. Do not leave any section empty in your final submission.

---

## 1. Study Title and Objectives

**Title:**
> *Fill in: A concise, descriptive title for your proposed study.*

**Primary Objective:**
> *Fill in: One sentence. What is the main clinical question your system answers about aortic valve replacement durability?*

**Secondary Objectives:**
> *Fill in: 2–4 additional goals (e.g., differentiating SAVR vs. TAVR failure trajectories, optimising echocardiographic surveillance intervals, reintervention/reoperation timing prediction).*

---

## 2. Target Population

### Inclusion Criteria
> *Fill in: Who qualifies for durability risk prediction? Consider valve type (bioprosthetic vs. mechanical), implant approach (SAVR/TAVR), age, minimum follow-up period, availability of serial echocardiographic data.*

### Exclusion Criteria
> *Fill in: Who should be excluded? Consider confounders (endocarditis, non-structural valve dysfunction, mechanical valve recipients, concurrent multi-valve disease, insufficient follow-up).*

### Cohort Size Estimate
> *Fill in: How large does the target cohort need to be for your proposed validation? Justify briefly, accounting for the relatively low annual incidence of structural valve deterioration.*

---

## 3. Endpoints

### Primary Endpoint
> *Fill in: What is the main outcome your model predicts (e.g., structural valve deterioration, hemodynamic valve deterioration, need for reintervention within N years)? How is it defined and measured? What performance threshold makes it clinically useful?*

### Secondary Endpoints
> *Fill in: List 2–4 secondary outcomes with their measurement approach (e.g., time-to-reintervention, mean gradient progression, paravalvular leak progression, valve thrombosis).*

| Endpoint | Measurement | Timeframe |
|---|---|---|
| | | |
| | | |

---

## 4. Proposed Data Sources

> *Fill in: List each data source your system would ingest. For each, describe: what data type it provides, how it would be accessed in a real deployment, and what specific variables are relevant.*

| Source | Data Type | Access Pathway | Key Variables |
|---|---|---|---|
| | | | |
| | | | |

**Ground Truth Definition:**
> *Fill in: How do you define a confirmed structural valve deterioration / durability failure event? What is your hierarchy (echocardiographic criteria per VARC-3/EAPCI-ESC consensus, reintervention record, explant pathology, clinical adjudication)? This is critical — be specific.*

---

## 5. Statistical Analysis Plan

### Sample Size
> *Fill in: How many event and non-event (or censored) cases do you need? Justify using event-per-variable (EPV) or a power calculation appropriate for time-to-event data.*

### Train / Validation / Test Split
> *Fill in: How will you partition data? How do you ensure no data leakage across patients or time? How do you handle the low base rate of SVD events (class imbalance)?*

### Evaluation Metrics
> *Fill in: Which metrics will you use to evaluate your model (e.g., time-dependent AUROC, C-index, calibration at fixed horizons)? Why are they appropriate given right-censored follow-up and the clinical cost of false negatives vs. false positives?*

### Subgroup Analyses
> *Fill in: Which patient subgroups will you evaluate separately (e.g., valve type/model, implant approach, age at implant, valve size)? Why?*

### Comparator / Baseline
> *Fill in: What is your model compared against? (e.g., STS-PROM risk score, published SVD nomograms, cardiologist gestalt, time since implant alone)*

---

## 6. Ethical Considerations and Data Privacy

### IRB / Ethics Review
> *Fill in: What ethical review would be required for this study? What exemptions might apply for retrospective de-identified data?*

### Data Privacy
> *Fill in: How do you ensure HIPAA compliance? How is PHI handled during model training and inference?*

### Algorithmic Fairness
> *Fill in: How will you evaluate and mitigate bias across demographic subgroups (age, sex, race/ethnicity) and across valve manufacturers/models?*

### Clinical Transparency
> *Fill in: How are model outputs presented to clinicians? What safeguards ensure the model is decision-support (e.g., prompting earlier echo surveillance) rather than decision-replacement for reintervention timing?*
