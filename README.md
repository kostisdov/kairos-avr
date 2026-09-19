<picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo/kairos-lockup-descriptor-dark.svg">
  <img src="logo/kairos-lockup-descriptor-bright.svg" alt="KAIROS" width="420">
</picture>

# Dynamic prediction of bioprosthetic aortic valve deterioration

Research prototype for the Dyania Health Hackathon 2026, AVR Durability Challenge.
*Kinetics of Aortic bioprosthesis Integrity, Risk-adapted echO Surveillance.*
From **καιρός**, the right moment: knowing which patients need to be looked at sooner.

> **Every model output in this repository is trained on synthetic scenarios and is illustrative and
> unvalidated.** Nothing here claims clinical accuracy, recommends reintervention or a treatment
> change, or alters guideline surveillance.

**Team kairos** · Panagiotis Sotiropoulos, physician · Kostis Dovelos, PhD, engineering ·
Minas Papageorgiou, engineering

| Deliverable | File |
|---|---|
| Study design | [`protocol/study_protocol.md`](protocol/study_protocol.md) |
| Model approach | [`model/approach.md`](model/approach.md) |
| Data plan | [`data/data_plan.md`](data/data_plan.md) |
| Slides | Coming soon |
| Proof of concept | [`notebooks/`](notebooks/) |

---

## The problem

Bioprosthetic aortic valves wear out, and they do it quietly. In a 1,387-patient surgical cohort
about a third developed haemodynamic deterioration over a decade, mostly without symptoms. In a
2,403-patient transcatheter registry, VARC-3 moderate or severe deterioration reached 10.8% at five
years.

Surveillance is a fixed schedule, the same for every patient and every valve. Roughly a third of
scheduled five-year echocardiograms in trials never happened, and echo readings vary between sites.
So deterioration is often recognised late, when reoperation is urgent rather than planned. In UK
national data on all reoperative aortic valve surgery from 1996 to 2019, operative mortality was
4.8% elective, 11.8% urgent and 31.7% emergency — a description of how presentation urgency and
mortality travel together across all reoperation causes, not a count of preventable deaths.

Every bioprosthetic recipient is affected. Most acutely, the growing group implanted under 65, who
will outlive their first valve, and for whom valve identity changes the odds materially.

**There is no durability prediction model in cardiac surgery today.** What exists is a fixed
schedule, a threshold read against the patient's own baseline echo, and symptoms. Current practice
asks whether a valve *is* failing. It never asks whether it *will*.

## Our approach

**Data.** What a hospital already holds: implant and procedure records, echo reports both structured
and free-text, laboratory results, dated medication history, and outcomes from procedure codes and
death records. Assembled per patient by text extraction and adjudicated against VARC-3 criteria.
Public device identifiers and guideline reference tables supply the device hierarchy and contextual
reference values; open-access trial reports and FDA device summaries supply simulation parameters.

**Method.** One primary model: penalised cause-specific proportional hazards for structural valve
deterioration, for death, and for non-SVD index-valve replacement, fitted on a landmark dataset so
that each prediction uses only information available at that time, and combined into cumulative
incidence. Biomarker and anticoagulant-exposure modules are added as separate blocks and judged on
incremental calibration and Brier score. Gradient boosting is an optional challenger, lower in
priority than the modules.

**What comes out.** The probability of deterioration before death at 1, 3 and 5 years plus a
12-month horizon, with the probability of death before deterioration and of remaining alive with an
intact valve, the main drivers, and a reliability statement. Three separate messages reach the
clinic: a current abnormality needing assessment now, a predicted risk high enough to bring the next
assessment forward, and a scheduled echo that has not happened. Guideline surveillance stays in
place; the model never recommends reintervention and never lengthens an interval.

## Three layers, kept apart

| Layer | Status | What it is |
|---|---|---|
| Real-record extraction | **Implemented** | A valve passport from 202 de-identified notes covering 117 patients: route, model and generation, size, implant year, serial gradients, event mentions. It establishes what a hospital record actually contains. **It does not train the model.** |
| Model development | **Demonstrated** | A penalised cause-specific model with dynamic landmarking, developed and evaluated on explicitly synthetic scenarios whose parameters are literature-informed, assumed, or varied in sensitivity analysis |
| Clinical validation | **Planned** | Retrospective development in adequate cohorts, a frozen model, temporal and external validation, then prospective silent evaluation. Its absence today is the boundary of a 36-hour event, not a failure of the objective |

## Key design decisions

| Decision | Rationale |
|---|---|
| Predict **adjudicated structural deterioration**, not generic valve failure. Death and non-SVD replacement compete; thrombosis is a reversible intercurrent state | Reintervention undercounts deterioration and arrives too late to change surveillance. A raised gradient is a sign, not a cause, so attribution is adjudicated. A valve replaced for endocarditis can never show structural deterioration, so that replacement competes; a treated thrombosis resolves, so it does not |
| **Time zero is the patient's own reference echo**, 30 to 180 days after implantation, not a fixed day-90 landmark | A day-90 prediction cannot use an echo performed on day 150. Anchoring on the actual study removes that contradiction and the look-ahead leakage that comes with it. The echo that establishes an endpoint is never used to predict that endpoint |
| The comparator that decides usefulness is **the guideline threshold rule**, not valve age and type | Valve age and type is a bar almost anything clears. What a cardiologist does today is read the current echo against the reference study. If the model cannot beat that, it adds nothing, whatever it scores |
| Biomarker and anticoagulant information as **separate modules on a working core**, each measured | Susceptibility biology and treatment history are where this differs from a valve-age chart, but plausibility is not prediction. Modular evaluation shows what each block adds and keeps the core usable when a marker is unmeasured |
| Irregular follow-up as an **observation process**; the three clinic messages kept separate | Patients with symptoms get more echoes and more detections. A missed visit triggers a reminder and lowers confidence; its association with deterioration is learned, not imposed. A five-year risk cannot justify a six-month appointment |
| Valve identity as **route and design class**, with model and generation where counts allow; partial pooling **planned, not claimed** | Model and generation differences are large and tangled with implant era. A frailty model pooling sparse devices toward their class is the right long-term structure, but the prototype uses the grouping it actually fits |

## Results

Four comparisons, all on explicitly synthetic scenarios and all reported including where the model
loses. Full records in [`docs/comparison/`](docs/comparison/).

| Comparison | Result |
|---|---|
| Against valve age and type, the weak reference | Gradual stenotic, SVD at 5 years, five-fold cross-validation at full size ([`docs/ladder_summary.md`](docs/ladder_summary.md)). The core model's Brier is **0.0354 against 0.0386** for valve age and type. The full model (core plus the biomarker and anticoagulant modules) has Brier 0.0350, area under the curve 0.74 (0.69 to 0.80) pooled over landmarks, and calibration slope 0.98. Without serial echocardiography the core model scores 0.0376, so **updating accounts for about two thirds of the gain** |
| Against the current gradient and its change (R4), the protocol's statistical comparator | Gradual stenotic, SVD at 5 years, five-fold cross-validation. KAIROS improves Brier by **0.0018 (95% CI 0.0007 to 0.0030)** and IPA from 0.060 to 0.105. The advantage is unchanged when the surveillance and visit-history terms are removed, so it does not come from how often patients are imaged. It holds on a temporal split (fitted on implants to 2016, scored on 2017 to 2020: AUC **0.78 against 0.73**) ([`docs/comparison/validation_splits/`](docs/comparison/validation_splits/)) |
| Against gradient boosting, the pre-specified challenger | **Retain the Cox model** at all three ladder steps. Mean Brier difference +5.6e-04, interval [-4.0e-06, +1.1e-03], spanning zero, on 10,701 rows from 2,390 patients. The challenger failed its own promotion rule and we publish that |
| Against current practice, the VARC-3 rule on a guideline schedule | Compared over time, as surveillance policies ([`docs/comparison/surveillance/`](docs/comparison/surveillance/)). With KAIROS allowed to bring the next echo forward at a 12-month SVD risk of 2%, deterioration is caught **1.8 months earlier on average (95% CI 1.2 to 2.6)** on a yearly schedule for **+2% echoes**. On the ACC/AHA calendar, which images surgical valves at 5 and 10 years, the share of deteriorating surgical valves caught before follow-up ends rises from **30% to 43%**. At the protocol's 5 to 15% band the policy barely changes, because baseline 12-month risk is about 0.3% |

Where the model is weakest, stated first: on abrupt regurgitant failure it barely improves on valve
age and type (Brier 0.0584 against 0.0601, area under the curve 0.65), because there is no preceding
gradient signal to read. The high-competing-mortality scenario has too few SVD events for the support
gates, so its metrics are withheld rather than reported; it is published as such rather than dropped.
The biomarker modules add almost nothing even where they carry signal by construction (0.0409 to
0.0407 in the meaningful-biomarker scenario).

## What we do not claim

No clinical prediction accuracy is demonstrated. No outcome improvement is shown. The prototype is
not deployment-ready. The supplied records carry year-only dates and almost no serial echo; they
establish extraction feasibility and descriptive counts, nothing more. Anticoagulant associations are
predictive, confounded by indication and era, and carry no treatment implication. To our knowledge
no externally validated, individual-level, dynamic prediction model of adjudicated structural valve
deterioration is in clinical use; existing work consists of cohort analyses of risk factors and
device comparisons.

---
## Layout

```
src/kairos/            core package (kairos-core)
  passport.py, varc3.py, km_reconstruct.py, build_*.py   existing data-plumbing layer (tested)
  extraction/          schema.py (all payloads), rules.py (span-aware rules), llm.py (Azure OpenAI, flag-gated)
  adjudication/        framework.py (reference study, candidates, endpoint, uncertain class)
  modelling/           landmark.py, features.py, cause_specific.py, cif.py, modules.py, predictor.py, train.py
  simulation/          scenarios.py (config/scenarios.yaml), generators.py (six synthetic scenarios)
  evaluation/          metrics.py (IPCW Brier, calibration, AUC), ladder.py, plots.py
  io/                  config.py (settings and the real-notes switch), storage.py (Blob or local), db.py
  privacy.py           privacy scan used by CI and the deploy scripts
services/              extract (FastAPI), predict (FastAPI), jobs (CLI), demo (Streamlit), one Dockerfile each
scripts/               deploy.ps1 / deploy.sh, privacy_scan.py, export_schemas.py, build_all.py
config/                scenarios.yaml (literature skeleton + simulation section), model.yaml, app.yaml
tests/                 unit, contract and service tests; synthetic note fixtures
docs/                  specification, design, runbook, deviations, JSON schemas
data/reference/        public reference tables (committed); data/derived/aggregates: small-cell-suppressed aggregates
```

Patient-level material (`*.xlsx`, `data/raw/`, `data/derived/private/`, `tmp/`) never enters
version control, a build context or a log. `scripts/privacy_scan.py` enforces this.

## Local quick start (Windows or Linux, Python 3.11)

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"        # Linux/macOS: .venv/bin/pip
.venv/Scripts/python -m pytest -q
.venv/Scripts/python scripts/build_all.py --quick   # scenarios, training, evaluation into artifacts/
```

Run the services locally (artefacts under `artifacts/`, SQLite database):

```bash
.venv/Scripts/python -m uvicorn app:app --app-dir services/extract --port 8001
.venv/Scripts/python -m uvicorn app:app --app-dir services/predict --port 8002
.venv/Scripts/python -m streamlit run services/demo/app.py
```

Or run the patient model alone, with both services in-process: set `KAIROS_DEMO_BACKEND=inprocess` and run only the
Streamlit command.

## Azure

One command provisions and deploys everything into `rg-kairos-dev` (Sweden Central):

```powershell
.\scripts\deploy.ps1
```

or `bash scripts/deploy.sh`, or `azd up` when the Azure Developer CLI is installed. The bicep
modules and the azd manifest are in `infra/` and `azure.yaml`; copy
`infra/main.bicepparam.example` to `infra/main.bicepparam` and fill in the four owner values
first. Details, the model-availability record, cost notes and the teardown are in
`docs/runbook.md` and `infra/README.md`.

## Interfaces

Both HTTP services publish OpenAPI documents (`/docs`). The JSON schemas of every payload
(passport, echo observation, exposure timeline, prediction, requests and responses) are
exported to `docs/schemas/` and checked for drift in CI.
