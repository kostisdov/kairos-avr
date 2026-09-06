# Data Plan

> **Instructions:** Describe the data your system would use in a real deployment, what you used for this prototype, and how you handle the gap between the two. Be honest about availability assumptions.

---

## 1. Data Sources — Real Deployment

> *Fill in: What data sources would your system ingest in a production clinical environment? For each, describe what it contains, how it would be accessed, and any integration dependencies.*

| Source | Data Type | Access Pathway | Variables Used |
|---|---|---|---|
| | | | |
| | | | |

---

## 2. Data Sources — Prototype / Hackathon

> *Fill in: What data did you actually use to develop and test your prototype? If you used public datasets or registries (e.g., PARTNER trial public data extracts, STS/ACC TVT Registry public reports, synthetic echo follow-up datasets), describe which subsets and why they are a reasonable proxy for the target population.*

| Dataset | Size Used | Why Selected | Limitations as Proxy |
|---|---|---|---|
| | | | |
| | | | |

---

## 3. Availability Assumptions

> *Fill in: What assumptions are you making about data availability that may not hold in every clinical site? (e.g., "Serial echocardiograms available for >60% of patients at 5-year follow-up", "Valve manufacturer/model consistently coded across sites"). Flag high-risk assumptions explicitly.*

---

## 4. Preprocessing and Data Quality

### Data Cleaning
> *Fill in: How do you handle duplicate records, out-of-range echo measurements, unit inconsistencies (e.g., mmHg vs. cmH2O), free-text echo reports?*

### Missing Data Strategy
> *Fill in: What is the expected missingness rate for your key features (e.g., missed follow-up echo visits)? How do you handle it (imputation method, missingness indicator features, exclusion thresholds)?*

### Temporal Alignment
> *Fill in: How do you handle echo measurements from different follow-up time points? What is your lookback/lookahead window relative to implant date? How do you select the "index" value for each feature at each prediction horizon?*

### Label / Ground Truth Construction
> *Fill in: How do you construct your training labels from raw echo/registry data? What is the prevalence of structural valve deterioration events in your dataset? How did you validate label quality against consensus criteria (e.g., VARC-3)?*

---

## 5. Synthetic or Proxy Data (if applicable)

> *Fill in: If you generated synthetic data or used simulated echo follow-up trajectories as proxy inputs, describe how. What limitations does this introduce? How would you replace synthetic data in a real study?*

---

## 6. Data Governance and Privacy

> *Fill in: What data use agreements (DUAs) cover your prototype datasets? What steps did you take to ensure no re-identification risk? How would data governance differ in a production deployment?*
