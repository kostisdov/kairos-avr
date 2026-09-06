# ML Approach

> **Instructions:** Describe your machine learning methodology. Be specific — vague answers score low. Every choice should have a justification tied to the clinical context.

---

## 1. Problem Formulation

> *Fill in: How did you frame the ML task? (e.g., binary classification at a fixed horizon, survival/time-to-event analysis, competing-risks model, longitudinal/sequence modelling of serial echo measurements). Why is this framing appropriate for predicting bioprosthetic aortic valve durability, given right-censored follow-up?*

---

## 2. Chosen Model(s)

> *Fill in: What model(s) did you select? List your primary model and any ensemble or secondary models.*

| Model | Role | Justification |
|---|---|---|
| | Primary | |
| | Baseline / comparator | |

> *Fill in: Why is your chosen model appropriate for this data type and clinical use case? Address: handling of censored/time-to-event outcomes, interpretability requirements, performance on longitudinal/tabular data, scalability to registry volumes.*

---

## 3. Feature Engineering

> *Fill in: What raw inputs does your model use? How do you transform them into model-ready features?*

### Input Variables
> *List the raw clinical variables your model ingests (e.g., baseline echo parameters, serial echo follow-ups, valve type/size/manufacturer, implant approach, patient-prosthesis mismatch, comorbidities, medications).*

### Engineered Features
> *List any derived or composite features (e.g., mean gradient progression rate, effective orifice area indexed to body surface area, patient-prosthesis mismatch flag, rate of change in peak velocity across follow-up visits). Explain why each adds signal.*

| Feature | Derivation | Clinical Rationale |
|---|---|---|
| | | |
| | | |

### Handling Missing Data
> *Fill in: What is your strategy for missing values, especially irregular or missed echocardiographic follow-up visits? (e.g., last-observation-carried-forward, model-native handling, missingness indicators). How do you flag informative missingness (e.g., a missed visit itself being a risk signal)?*

---

## 4. Validation Strategy

> *Fill in: How do you ensure your model generalises and does not overfit to your training cohort?*

- **Train / Validation / Test split:**
- **Cross-validation approach:**
- **Temporal validation** (if applicable — training on earlier implant cohorts, testing on later ones):
- **External validation** (if applicable — held-out site, registry, or valve manufacturer cohort):

---

## 5. Expected Model Outputs

> *Fill in: What does a single model inference return? (e.g., probability of structural valve deterioration at 5/8/10 years, risk tier, ranked feature contributions, recommended next surveillance interval). How is this output consumed by a clinician or downstream system?*

**Output format:**
> *e.g., "Time-dependent probability of SVD at 5/8/10 years + risk tier (Low / Moderate / High) + top-3 contributing features via SHAP + recommended echo follow-up interval"*

---

## 6. Clinical Integration

> *Fill in: Where does this model sit in the clinical workflow? (e.g., passive EHR/registry flag, echo surveillance scheduling support, structural heart team referral trigger, trial recruitment filter for reintervention studies). What triggers a model inference? What action does a High-risk flag prompt?*

---

## 7. Limitations and Failure Modes

> *Fill in: Where does your model break down? What patient profiles, valve types, or data quality scenarios (e.g., sparse follow-up, inter-observer echo variability, new valve models with limited historical data) lead to unreliable outputs? How should clinicians be informed of these?*
