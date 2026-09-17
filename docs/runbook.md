# KAIROS runbook

Operating guide for the KAIROS research prototype on Azure and on a local machine. Written
16 September 2026 against `docs/kairos_azure_build_design.md`. Every model output of this
system is trained on synthetic scenarios and is illustrative and unvalidated.

## 1. Status at the end of the build session (16 September 2026)

| Milestone | State | Evidence |
|---|---|---|
| M0 Scaffold | Repository layout, Bicep, `azure.yaml`, CI and privacy scan are in place; `az bicep build` and `az deployment sub what-if` succeed (33 resources to create). **`azd up` / the deploy script has not been run**: the build session's permission mode blocked the infrastructure-apply command, so the owner runs it once (section 4) | `infra/`, `scripts/deploy.*`, `.github/workflows/` |
| M1 Core | Landmark builder with leakage rules, penalised cause-specific model, CIF combination, metrics; unit tests for leakage, endpoint-echo exclusion and probability sums; runs on every scenario | `tests/test_landmark_leakage.py`, `tests/test_cif_metrics.py`, `tests/test_model_pipeline.py` |
| M2 Extraction service | Rules path implemented and tested; hosted-model path behind the flag with the refusal tested (HTTP 403 with real text and the flag off); field-level accuracy on fixtures produced by the `extraction-eval` job (rules only so far; the gpt-5.1 comparison runs once the deployment exists); request parameters pinned by `tests/test_llm_client.py` | `tests/test_extract_service.py`, `artifacts/metrics/latest/extraction_fixtures_summary.csv` |
| M3 Jobs | Scenario, train and evaluate commands implemented as one image with three Container Apps Jobs; manifests carry seed, git sha and config hash; run locally into `artifacts/` | `services/jobs/cli.py`, `artifacts/` |
| M4 Predict service | Contract tests against the schema, 409 on endpoint-met passports, labels in every response | `tests/test_predict_service.py` |
| M5 Demo | Interactive patient model: every model input editable (valve, patient at implant, echo series with live VARC-3 staging against the reference study, biomarker values by module, antithrombotic exposure history), six synthetic example patients plus the train job's published patient, prediction with the four outcome probabilities, drivers and reliability, the three separated messages beside the unchanged guideline schedule, risk at every echo, a what-if for the next echo, and filling the form from a note. Verified locally by `tests/test_demo_ui.py` and in the browser; Entra sign-in takes effect only after deployment | `services/demo/app.py`, `src/kairos/demo_patient.py` |
| M6 Hardening | This runbook, `docs/deviations.md`, model-availability record (section 3), privacy scan in CI; the cost record (section 8) is to be filled after the first week in Azure | `docs/` |

Test suite: 149 tests pass (the original 39 plus 110 new), `ruff check .` is clean, the
privacy scan is clean in repository mode, and the payload schemas in `docs/schemas/` match
the code.

## 2. Environment record

- Build machine: Windows 11, Python 3.11.6, Azure CLI 2.87.0 with Bicep 0.47.16, Git
  2.52, GitHub CLI 2.96. **No Docker, no `azd`.** Images are therefore built by Azure
  Container Registry (`az acr build`) and infrastructure is deployed with
  `az deployment sub create`; this is the reference path (design section 6a). `azure.yaml`
  is provided for `azd up` on a machine that has `azd`.
- Subscription `004a53c3-3a7f-4d2f-b288-483265274096` ("Azure subscription 1"), tenant
  `b38e34be-d9fb-457f-b1b8-d4d3b2ef89fb`, signed in as the owner (Owner role at
  subscription scope). A second subscription in another tenant is visible; the scripts
  set the subscription explicitly.
- Region: Sweden Central for everything. Resource providers were registered on
  16 September 2026 (`Microsoft.App` was not registered before; Container Apps quota
  in the region: 15 managed environments).
- Python environment: `.venv` created from `pyproject.toml`; `requirements.lock.txt`
  pins the resolved versions for the container images.
- The repository directory is **not** a git repository. `model_version` therefore ends in
  `+nogit` until the owner runs `git init`; the deploy script tags images with a UTC
  timestamp in that case and with the short commit hash once git exists.

## 3. Model availability record (Azure OpenAI, Sweden Central, 16 September 2026)

Account: `papageorgiouminas-0092-resource` (kind AIServices, resource group
`rg-papageorgiou.minas-2513`, endpoint `https://papageorgiouminas-0092-resource.openai.azure.com/`,
system-assigned identity, local auth enabled, public network access enabled). Existing
deployments on the account (never touched by KAIROS): `metaklisi-gpt-5-nano`,
`metaklisi-gpt-5-mini`, `mdm-reader-luna`, `mdm-reader-terra`.

`az cognitiveservices account list-models` (filtered to the candidates of design section 4):

| Model | Version | Lifecycle | Deployable SKUs listed | Quota in this subscription (`az cognitiveservices usage list -l swedencentral`) |
|---|---|---|---|---|
| gpt-4.1 | 2025-04-14 | Legacy (retirement 2027-04-14) | Standard, GlobalStandard, DataZoneStandard, batch, provisioned | **none** for Standard, GlobalStandard or DataZoneStandard (only GlobalBatch and DataZoneBatch entries exist) |
| gpt-4.1-mini | 2025-04-14 | Legacy | as above | **none** except batch |
| gpt-5.1 | 2025-11-13 | GenerallyAvailable (retirement 2027-05-15) | **Standard** (the only GPT-5 model with a regional SKU), GlobalStandard, DataZoneStandard, provisioned, batch | Standard 3000, GlobalStandard 30000, DataZoneStandard 10000 |
| gpt-4o | 2024-11-20 | Legacy (retirement 2027-04-14) | Standard, GlobalStandard, DataZoneStandard, batch, provisioned | Standard 1000, GlobalStandard 30000, DataZoneStandard 10000 (units as reported by the CLI, thousands of tokens per minute) |
| gpt-4o | 2024-08-06 | Deprecating (2027-04-14) | as above | same pool as gpt-4o |
| o4-mini | 2025-04-16 | Deprecating (retirement 2026-11-19) | Standard, GlobalStandard, DataZoneStandard, batch | Standard 10000, GlobalStandard 10000, DataZoneStandard 3000 |
| gpt-5, gpt-5-mini, gpt-5.1, gpt-5.4, gpt-5.4-mini, gpt-5.5, gpt-5.6-* | 2025-08 to 2026-07 | GenerallyAvailable | GlobalStandard, DataZoneStandard (regional Standard only for gpt-5.1) | quota present |
| Llama-3.3-70B-Instruct, Mistral-Large-3 | | GenerallyAvailable | GlobalStandard (DataZoneStandard for Mistral) | quota present |

Decision, revised 16 September 2026 at the owner's request ("use a better gpt model, it's ok
costwise for the demo for the next 2 days"): extraction and adjudication assistance both use
**`gpt-5.1` version `2025-11-13`** on the regional **`Standard`** SKU, deployments
`kairos-extract` (capacity 100, i.e. 100 000 tokens per minute, reasoning effort `low`) and
`kairos-adjudicate` (capacity 50, reasoning effort `medium`). A re-check the same evening
confirmed that `gpt-5.1` is the only GPT-5 model offered as a regional Standard deployment in
Sweden Central, with 3 000 capacity units of quota. `gpt-5.4`, `gpt-5.5` and `gpt-5.6` exist
only as Global or EU Data Zone deployments, which may process note text outside Sweden
Central, so they were not used (design section 4 requires deployments in the region of the
data). The pre-screen stays **rules only**. The first decision of the day, `gpt-4o`
`2024-11-20` for extraction and `o4-mini` for adjudication, is superseded.

Capacity is set well above the demo's needs because Azure counts each request's
`max_completion_tokens` (6 000 for extraction, 8 000 for adjudication, reasoning included)
against the per-minute limit. Standard deployments are billed per token, so capacity costs
nothing while idle.

GPT-5.1 is a reasoning model. The services send `reasoning_effort` and no temperature, with
the API version `2025-04-01-preview` (setting `v1` switches to the version-free
`/openai/v1/` endpoint). The deployments are created by `infra/openai.bicep` when the owner
deploys. The build session could neither create nor call them: the permission system
denied `az cognitiveservices account deployment create` on the shared account. The request
parameters are therefore pinned by unit tests and verified live by `llm-check` (section 4).

Abuse monitoring: the default abuse-monitoring configuration of the account was **not
verified** in this session (it is not exposed by the CLI commands used). Azure OpenAI does
not train on customer data; the application never logs prompts or completions. If the
owner wants the modified-abuse-monitoring exemption, file the request from the Azure AI
Foundry portal for the account and record the outcome here.

Data residency: `Standard` SKU deployments process in Sweden Central. Every call carries the
header `x-kairos-source: real|synthetic`; the managed identity holds *Cognitive Services
OpenAI User* on the account; no API key is stored anywhere.

## 4. Deploying to Azure (owner, one command)

Prerequisites: `az login` as the owner; Python 3.11 on the path (the scripts run the
privacy scan and parse deployment outputs).

```powershell
.\scripts\deploy.ps1
```

or from Git Bash:

```bash
bash scripts/deploy.sh
```

What the script does, in order (idempotent, safe to re-run):

1. Sets the subscription and registers the resource providers.
2. `az deployment sub create` with `infra/main.bicep` and `infra/main.bicepparam`:
   resource group `rg-kairos-dev`, managed identity, Log Analytics and Application
   Insights, Container Registry, Storage (seven containers), Key Vault, VNet with two
   delegated subnets, private DNS zone, PostgreSQL Flexible Server (Entra-only, private
   access), the Container Apps environment, three apps and three jobs with the placeholder
   image, the monthly budget (150 USD, alerts at 80 % actual and 100 % forecast), and the
   two model deployments on the existing Azure OpenAI account. Outputs go to
   `infra/outputs.local.json` (git-ignored). Expect 15 to 25 minutes, mostly PostgreSQL
   and the environment.
3. Creates the Entra app registration `kairos-demo-dev` for the demo login, requires
   assignment, assigns the owner (add teammates with `-AllowedUserObjectIds id1,id2`
   once they are guests in the tenant), and stores a client secret in Key Vault once.
4. Stages an allow-listed build context (never the spreadsheets, `data/raw`,
   `data/derived/private` or `tmp`), runs the privacy scan on it, and builds the four images
   in the registry with `az acr build`.
5. Re-runs the deployment with the image references and the demo client id.
6. Prints the demo URL and the internal service URLs.

Confirm the model deployments answer before anything else (about 30 seconds; one short
synthetic structured-output call per deployment, signed in as the owner):

```powershell
$env:KAIROS_OPENAI_ENDPOINT = "https://papageorgiouminas-0092-resource.openai.azure.com/"
$env:KAIROS_OPENAI_ADJUDICATE_DEPLOYMENT = "kairos-adjudicate"
.venv\Scripts\python services\jobs\cli.py llm-check
```

Every role should report `"ok": true` (the pre-screen reports `null`, not configured). If
a call fails with an API-version or parameter error, run `llm-check --api-version v1`; when
that passes, set `openAiApiVersion = 'v1'` in `infra/main.bicepparam` and re-run the deploy
script with `-SkipBuild -SkipAuth`. The same check runs inside Azure with
`az containerapp job start -n kairos-job-scenarios -g rg-kairos-dev --args llm-check`.

Then run the jobs, in this order (each writes to the storage account and records a run in
PostgreSQL):

```bash
az containerapp job start -n kairos-job-scenarios -g rg-kairos-dev
az containerapp job start -n kairos-job-train -g rg-kairos-dev
az containerapp job start -n kairos-job-evaluate -g rg-kairos-dev
az containerapp job start -n kairos-job-scenarios -g rg-kairos-dev --args upload-reference
az containerapp job start -n kairos-job-scenarios -g rg-kairos-dev --args extraction-eval
```

Watch executions with `az containerapp job execution list -n <job> -g rg-kairos-dev -o table`
and logs in Log Analytics (`ContainerAppConsoleLogs_CL`). The predict service loads
`models/latest/bundle.joblib` at start; after the first train job restart it or call
`POST /reload` from inside the environment (the demo can trigger a restart by scaling).
Database tables are created on first use by every service; no separate migration is needed
(`--args migrate` exists for an explicit check).

Redeploy code only: `.\scripts\deploy.ps1 -SkipAuth` (re-builds images and re-wires them).
GitHub Actions (`.github/workflows/deploy.yml`) does the same through OIDC once the
federated credential in `infra/README.md` exists; no secret is stored in GitHub.

To switch off the real-notes path in one place:

```bash
az containerapp update -n kairos-extract -g rg-kairos-dev --set-env-vars ALLOW_REAL_NOTES_TO_LLM=false
```

(or set `allowRealNotesToLlm = false` in `infra/main.bicepparam` and re-deploy). The flag is
read from configuration only; a request cannot change it.

## 5. Running locally

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q                       # 222 tests
.venv/Scripts/python scripts/build_all.py --quick        # scenarios, train, evaluate, extraction-eval into artifacts/ (quick namespace)
.venv/Scripts/python services/jobs/cli.py scenarios --all --namespace full               # configured cohort sizes
.venv/Scripts/python services/jobs/cli.py scenarios --all --n 600 --namespace quick      # any other size
.venv/Scripts/python scripts/label_policy_report.py --namespace full                     # SVD events under each label policy
.venv/Scripts/python scripts/cif_convention_report.py --namespace full                   # old versus current probability integration
.venv/Scripts/python services/jobs/cli.py evaluate --all --quick --namespace quick       # four-state ladder, quick gates (exploratory)
.venv/Scripts/python services/jobs/cli.py evaluate --scenario irregular_surveillance --quick --namespace quick --sensitivity observation
.venv/Scripts/python services/jobs/cli.py size-pilot --all --n 2000 --dev-seeds 3        # writes config/evaluation_plan.yaml, unfrozen
.venv/Scripts/python services/jobs/cli.py train --family both                           # Cox replaces models/latest; boosting goes to models/<version>/gradient_boosting/
.venv/Scripts/python services/jobs/cli.py evaluate --scenario gradual_stenotic --quick --namespace quick --families cox gradient_boosting
.venv/Scripts/python services/jobs/cli.py evaluate ... --run-id <run>                   # resume: completed folds load from metrics/runs/<run>/oof/
.venv/Scripts/python services/jobs/cli.py compare --run-id <run> --scenario gradual_stenotic --step core_plus_both
.venv/Scripts/python services/jobs/cli.py freeze-candidate --run-id <run> --scenario gradual_stenotic --step core_plus_both
.venv/Scripts/python services/jobs/cli.py evaluate-final --run-id <run>                 # once, frozen plan only
.venv/Scripts/python scripts/boosting_benchmark.py --tune                                # compute projection for the comparison
.venv/Scripts/python services/jobs/cli.py calibrate-generator --population synthetic_recovery --free death_annual_at_79,p_diabetes --synthetic-recovery "death_annual_at_79=0.15,p_diabetes=0.42" --n 2000 --budget 60
.venv/Scripts/python services/jobs/cli.py calibrate-generator --evidence data/reference/evidence_registry/candidates_published_rates.yaml --population external_population_notion_like
.venv/Scripts/python services/jobs/cli.py generate-calibrated --bundle <cal-id> --n 2500 --replicates 10 --seed 3201
.venv/Scripts/python services/jobs/cli.py validate-cohort --cohort calibrated/<cal-id>/quick/fit-seed3201
.venv/Scripts/python services/jobs/cli.py train-calibrated --cohort calibrated/<cal-id>/quick/fit-seed3201   # models/experimental only
.venv/Scripts/python -m uvicorn app:app --app-dir services/extract --port 8001
.venv/Scripts/python -m uvicorn app:app --app-dir services/predict --port 8002
.venv/Scripts/python -m streamlit run services/demo/app.py
```

Cohorts (generator 2.0, endpoint version 2, from 17 September 2026) are stored under
`scenarios/<scenario>/<variant>/<quick|full>/<tag>/`, with the evaluation-only truth tables in a
`truth/` subfolder, and the pointer is `scenarios/<scenario>/<variant>/<namespace>/latest.json`.
`train` and `evaluate` take `--namespace` (default `full`). Cohorts written by generator version 1
(pointer `scenarios/<scenario>/<variant>/latest.json`) are not read by the jobs: run the scenarios
job again before training, locally and in Azure. A full-namespace run refuses a size other than
the configured `n_patients`.

Evaluation (Phase B, 17 September 2026). Every ladder row is per state (SVD, death, non-SVD replacement,
alive intact), horizon and weighting (pooled landmark rows; patient balanced), with support status,
coverage, bootstrap validity and the integration and endpoint versions. Quick-namespace results use the
relaxed quick gates and are exploratory by construction. The evaluation plan (`config/evaluation_plan.yaml`,
written by the sizing pilot) is advisory: every evaluation runs and records `plan_status` (`frozen_match`,
`unfrozen_plan`, `no_plan`, `plan_mismatch` or `quick`); only a full run matching a frozen plan is labelled
`prespecified` evidence, everything else `exploratory` (owner decision, 17 September 2026: the data are synthetic). Model
bundles trained before the hazard contract are rejected at load (`LegacyBundleError`): retrain with
`cli.py train`. A request whose route is missing or unsupported returns HTTP 422; no route is substituted.

Model families (Phase C, 17 September 2026). `evaluate --families cox gradient_boosting` tunes boosting on
patient-grouped inner folds of every outer training set, writes both ladders, and for each step a paired
comparison on common rows plus a promotion decision (`metrics/<version>/<scenario>/comparison/`). Quick or
unfrozen runs still get a promote/retain decision, labelled `evidence: exploratory`. Bundles are schema 2; retrain after upgrading scikit-learn,
lifelines or scikit-survival (major.minor), otherwise the service refuses to load them.

Calibrated generator (17 September 2026, `docs/kairos_calibrated_generator_design.md`). `kairos.calibration` fits a small
declared parameter set (`config/calibration/parameters_v1.yaml`) to reviewed, compatible targets for a population
(`config/calibration/populations.yaml`) and writes an immutable bundle under `calibration/<bundle_id>/` with a report.
Evidence candidates from `published_rates.csv` are in `data/reference/evidence_registry/candidates_published_rates.yaml`,
all `proposed`: to use one, set `review_status: approved`, a `role` (fit or holdout), an uncertainty and
`acceptance.tolerance`, and resolve the compatibility note (endpoint definition, horizon, 1-KM vs cumulative incidence).
Until then real calibrations report `insufficient_evidence`. Generation requires an accepted bundle; calibrated cohorts live
under `scenarios/calibrated/<bundle>/<namespace>/<replicate>/` and models trained on them only under `models/experimental/`.
`profile-observed` needs a published, frozen normalized research snapshot (standardization service, not yet built).

To run the demo without starting the two services, set `KAIROS_DEMO_BACKEND=inprocess`: the page then runs the
extract and predict apps inside its own process and loads `artifacts/models/latest/bundle.joblib`.

Locally the artefact store is `artifacts/` and the database is `artifacts/kairos.db`
(SQLite). Setting `KAIROS_OPENAI_ENDPOINT` and `KAIROS_OPENAI_API_KEY` (local only) enables
the hosted-model path against the same deployments; in Azure the managed identity is used.

The data-plumbing layer (`src/kairos/build_*.py`, `passport.py`) reads the de-identified
spreadsheets and runs only on the owner's machine; `scripts/build_all.py` skips it when the
spreadsheets are absent.

## 6. Privacy controls

- `.gitignore` and `.dockerignore` exclude `*.xlsx`, `data/raw/`, `data/derived/private/`, `tmp/`.
- `scripts/privacy_scan.py` (module `kairos.privacy`) fails CI, the deploy scripts and the
  reference upload job on `Patient_###` identifiers, spreadsheets, forbidden paths, source
  spreadsheet column names in data files, and long clinical free text in data files.
- The extract service stores extracted fields and short evidence snippets, never note
  text; the logging guard truncates any log line above 400 characters and drops records
  tagged with note text; Azure Monitor sampling is on.
- The real-notes comparison on the 202 notes (design section 4) is a local-only procedure:
  run the extract service locally with the hosted-model path configured, feed the notes
  from the spreadsheet through `/extract` with `source_kind=real` and `store=false`, and
  write only the aggregate agreement table under `data/derived/aggregates/`. It was not run
  in this session (no deployment existed yet).

## 7. Results of the local run (synthetic scenarios, illustrative)

Written by `services/jobs/cli.py` into `artifacts/` on 16 September 2026 with
`--n 900` cohorts, 3-fold patient-grouped cross-validation and 30 bootstrap resamples
(`--quick`); the results are in section 10. The full-size run (2 500 patients, 5 folds, 200 resamples) is what the Azure
evaluate job produces. See `artifacts/metrics/latest/ladder_all.csv` and
`artifacts/figures/latest/`. A summary is appended in section 10 once the run completes.

## 8. Cost record

Estimate before the first bill (design section 6): PostgreSQL B1ms with 32 GiB about
13 USD per month, Container Registry Basic about 5 USD, Log Analytics at pay-as-you-go
under 1 GB per day, storage a few cents, Container Apps at scale-to-zero close to zero when
idle, model tokens a few USD per hundred thousand tokens. The Cost Management budget on
`rg-kairos-dev` is 150 USD per month with alerts at 80 % actual and 100 % forecast.

| Date | Resource group spend to date (USD) | Source |
|---|---|---|
| to be filled after the first week | | Cost Management, `rg-kairos-dev` |

## 9. Teardown

```bash
az group delete -n rg-kairos-dev --yes
az cognitiveservices account deployment delete -n papageorgiouminas-0092-resource -g rg-papageorgiou.minas-2513 --deployment-name kairos-extract
az cognitiveservices account deployment delete -n papageorgiouminas-0092-resource -g rg-papageorgiou.minas-2513 --deployment-name kairos-adjudicate
az ad app delete --id "$(az ad app list --display-name kairos-demo-dev --query '[0].appId' -o tsv)"
```

The role assignment of the deleted identity on the OpenAI account disappears with the
identity. Nothing else outside the resource group is created.

## 10. Local run summary

Run of 16 September 2026, 23:14 to 23:38, after fixing the horizon-censoring bug in the IPCW
metrics (rows capped at the 5-year horizon were treated as lost instead of event-free; now
covered by a regression test). Cohorts of 900 requested patients, 3-fold patient-grouped
cross-validation, 30 patient-level bootstrap resamples. Synthetic scenarios only:
illustrative, unvalidated.

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

### Reading the results

- **Serial echo adds information where the scenario contains it.** In the gradual stenotic
  scenario the core model beats both the valve-age-and-type reference and the core model
  without serial echo, with an AUC of 0.80 at 5 years.
- **Abrupt regurgitant failure stays hard to predict, as designed.** The core model is no
  better than the reference, and the calibration slope drops to 0.67.
- **The high competing mortality scenario yields only 3 SVD events at 5 years.** Its SVD
  metrics are not interpretable. The probability bookkeeping it is meant to exercise is
  covered by the tests instead. Open point: the assumed death multiplier makes death about
  twenty times as likely as SVD, not several times; lower it in `config/scenarios.yaml`.
- **The biomarker modules show no incremental value, even in the meaningful variant.** The
  renal and metabolic markers already sit in the core, as the specification places them.
  Phosphate and Lp(a) carry the module-specific signal but are measured in only 30 to 40 %
  of rows. Open point: raise their measured fraction or effect size in the meaningful variant.
- **The anticoagulant block recovers opposite directions but adds no Brier gain.** In the
  full-data fit, current VKA exposure raises the SVD hazard and any current anticoagulation
  lowers the thrombosis hazard. These are predictive associations in synthetic data only.
- **dp-ucMGP (17 September run, `evaluate --scenario anticoagulant_mechanism_confounding --steps core core_plus_anticoagulant`,
  5 folds, 200 patient bootstraps, seed 20260916).** Substudy of about half the patients; offset model on measured rows only.

  | variant | test of dp-ucMGP terms (patient-bootstrap Wald p) | VKA contrast without → with marker (shrinkage) | Δ Brier at 5 y, measured rows (95% CI) |
  |---|---|---|---|
  | marker_mediated | 0.009 | 0.455 → 0.169 (63 %) | −0.0011 (−0.0027 to 0.0001) |
  | marker_noise | 0.007 | 0.658 → 0.690 (−5 %) | −0.0006 (−0.0017 to 0.0005) |

  The VKA contrast falls toward zero only when the marker carries the signal, as specified. The incremental Brier
  gain is not distinguishable from zero in either variant. The noise variant's significant test at this seed is driven by the
  interaction term (−0.27, SE 0.095). On generator seeds 12, 13 and 14 the noise variant gave p = 0.31, 0.36 and 0.28, and the
  mediated variant gave p = 0.0003, 0.013 and 0.0003, so we read it as a chance finding at one seed rather than a
  systematic artefact. Synthetic scenarios only.
- **Calibration is within the 3-point usefulness threshold in every interpretable
  scenario.** The model over-predicts by 0.7 to 2.9 percentage points at 5 years; slopes
  run from 0.64 to 0.93.
- **Extraction on the synthetic fixtures:** the rules path was correct on route, model and
  size for all 16 labelled fixtures. The hosted-model comparison needs the Azure
  deployment and has not been run.


## 11. Publish and rollback (run-scoped model store)

`train` writes each family to `models/runs/{run_id}/{family}/` (`bundle.joblib`, `card.json`,
`demo_patient.json`) and prints the run id and each family's support decisions. It never changes
what the predict service serves. Serving is controlled by one pointer per namespace and family,
`models/{namespace}/{family}/active.json` (`run_id`, `path`, `sha256`, `previous`, `published_at`).
`models/latest/` is still written by `train` for the legacy loader only.

```bash
# 1. train both families (prints run_id and support decisions; publishes nothing)
python services/jobs/cli.py train --family both --namespace full

# 2. evaluate, compare, freeze and final-test before publishing (see the detailed design, WP-D4)
python services/jobs/cli.py evaluate --all --families cox gradient_boosting
python -m pytest -m slow

# 3. publish a run for one family (prints the new pointer; the old one is kept as `previous`)
python services/jobs/cli.py publish --family cox --run-id <run_id> --namespace full
python services/jobs/cli.py publish --family gradient_boosting --run-id <run_id> --namespace full

# 4. roll back to the previous pointer (prints the restored pointer)
python services/jobs/cli.py publish --family cox --rollback --namespace full
```

- Publishing a run that has no bundle for that family fails with exit code 2 (nothing is changed).
- Rolling back without a previous pointer fails with exit code 2. One level of history is kept; a
  second rollback swaps back to the pointer that was just replaced.
- The service verifies the bundle's SHA-256 against the pointer when loading and refuses a mismatch.
  Restart (or redeploy) the predict service after a publish or rollback so it reloads the pointers.
- Quick-namespace runs (`build_all.py --quick`) are stamped `smoke test: not evidence` in their
  cards, demo patients and `run.json`; publish them only to the `quick` namespace.
