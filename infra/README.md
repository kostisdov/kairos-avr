# KAIROS on Azure: infrastructure

Bicep templates and deployment scripts for the KAIROS research prototype
(`docs/kairos_azure_build_design.md`, sections 6 to 9). Everything lands in one
resource group (`rg-kairos-dev`) in Sweden Central, plus two model deployments on the
owner's existing Azure OpenAI account. Managed identity everywhere; no API keys, no
secrets in files.

| File | Purpose |
|---|---|
| `main.bicep` | Subscription-scoped entry point: resource group, then the three modules below; all outputs |
| `identity.bicep` | User-assigned managed identity `id-kairos-<env>` (created first so its object id can feed the other modules) |
| `openai.bicep` | Deployments `kairos-extract` and `kairos-adjudicate` (both `gpt-5.1`, regional Standard) on the existing account `papageorgiouminas-0092-resource` (resource group `rg-papageorgiou.minas-2513`), created one at a time, plus the *Cognitive Services OpenAI User* role for the identity. The account and its other deployments are never modified |
| `resources.bicep` | Log Analytics, Application Insights, Container Registry, Storage, Key Vault, VNet, private DNS zone, PostgreSQL Flexible Server, Container Apps environment, three apps, three jobs, budget, role assignments |
| `main.bicepparam` | Owner defaults for `dev` |
| `outputs.local.json` | Written by the deploy scripts (git-ignored); the deployment outputs as JSON |
| `../azure.yaml` | Azure Developer CLI manifest (optional path) |
| `../scripts/deploy.ps1`, `../scripts/deploy.sh` | Reference deployment path with the plain Azure CLI |
| `../.github/workflows/deploy.yml` | Same script from GitHub Actions with OIDC |

## Parameters (`main.bicep`)

| Parameter | Default | Meaning |
|---|---|---|
| `environmentName` | `dev` | Suffix of the environment-scoped names, `KAIROS_ENV`, tags `environment` and `azd-env-name` |
| `location` | `swedencentral` | Region of every resource |
| `resourceGroupName` | `rg-kairos-dev` | Resource group created by the template |
| `ownerEmail` | `''` (`main.bicepparam`: owner e-mail) | `owner` tag and budget notifications; empty skips the budget |
| `deployerObjectId` | `''` (`main.bicepparam`: owner object id) | Gets *Storage Blob Data Contributor* on the storage account and *Key Vault Secrets Officer* on the vault; empty skips. The scripts override it with the signed-in user |
| `postgresAdminObjectId` | `''` (`main.bicepparam`: owner object id) | Second PostgreSQL Entra administrator (type `User`); empty skips |
| `postgresAdminPrincipalName` | `''` (`main.bicepparam`: owner UPN) | Principal name of that user |
| `openAiAccountName` | `papageorgiouminas-0092-resource` | Existing Azure AI Services account |
| `openAiResourceGroup` | `rg-papageorgiou.minas-2513` | Its resource group |
| `openAiApiVersion` | `2025-04-01-preview` | `KAIROS_OPENAI_API_VERSION`; `v1` switches the services to the version-free `/openai/v1/` endpoint |
| `openAiExtractDeploymentName` / `ModelName` / `ModelVersion` / `SkuName` / `Capacity` | `kairos-extract` / `gpt-5.1` / `2025-11-13` / `Standard` / `100` | Extraction deployment (regional, pay-per-token; capacity is throughput only) |
| `openAiAdjudicateDeploymentName` / `ModelName` / `ModelVersion` / `SkuName` / `Capacity` | `kairos-adjudicate` / `gpt-5.1` / `2025-11-13` / `Standard` / `50` | Adjudication-assist deployment |
| `openAiExtractReasoningEffort` / `openAiAdjudicateReasoningEffort` | `low` / `medium` | `KAIROS_OPENAI_EXTRACT_REASONING_EFFORT`, `KAIROS_OPENAI_ADJUDICATE_REASONING_EFFORT`; empty for a non-reasoning model, which then receives temperature 0 |
| `openAiPrescreenDeploymentName` / `ModelName` / `ModelVersion` / `SkuName` / `Capacity` | `kairos-prescreen` / `''` / `''` / `Standard` / `10` | Pre-screen deployment; an empty model name skips it and `KAIROS_OPENAI_PRESCREEN_DEPLOYMENT` stays empty (rules-only pre-screen) |
| `extractImage`, `predictImage`, `demoImage`, `jobsImage` | `''` | Full image references (`<acr>.azurecr.io/kairos-<svc>:<tag>`); empty means the placeholder `mcr.microsoft.com/k8se/quickstart:latest` |
| `demoAuthClientId` | `''` | Application id of `kairos-demo-<env>`; empty leaves the demo without built-in authentication |
| `allowRealNotesToLlm` | `true` | Rendered as `ALLOW_REAL_NOTES_TO_LLM` (`true`/`false`); design rule 0.3 |
| `budgetAmount` | `150` | Monthly budget on the resource group |
| `budgetStartDate` | `utcNow('yyyy-MM-01')` (`main.bicepparam`: `2026-09-01`) | First of a month; pinned in the parameter file so re-deployments do not rewrite the budget |
| `gitSha` | `''` | `KAIROS_GIT_SHA`; the scripts pass the image tag |

Outputs: `resourceGroupName`, `identityName`, `identityClientId`, `identityPrincipalId`,
`acrName`, `acrLoginServer`, `storageAccountName`, `storageAccountUrl`, `keyVaultName`,
`keyVaultUrl`, `postgresFqdn`, `postgresServerName`, `postgresDatabaseName`,
`containerAppsEnvironmentName`, `environmentDefaultDomain`, `demoFqdn`,
`extractInternalUrl`, `predictInternalUrl`, `openAiEndpoint`,
`appInsightsConnectionString`, and the effective `extractImage`, `predictImage`,
`demoImage`, `jobsImage`, `demoAuthClientId` (so a re-run can preserve them).

## What gets created

Names with `<u>` carry `uniqueString(resourceGroup().id)`.

- Identity `id-kairos-<env>`: AcrPull, Storage Blob Data Contributor, Key Vault Secrets User, Cognitive Services OpenAI User, PostgreSQL Entra administrator.
- `log-kairos-<env>` (PerGB2018, 30 days, 1 GB/day cap) and `appi-kairos-<env>`.
- `crkairos<u>` (Basic, admin user off).
- `stkairos<u>` (StorageV2, LRS, hierarchical namespace, no public blobs, shared keys off, TLS 1.2, RBAC only) with containers `scenarios`, `models`, `metrics`, `figures`, `reference`, `aggregates`, `raw`.
- `kv-kairos-<u>` (RBAC, soft delete 7 days, purge protection off).
- `vnet-kairos-<env>` 10.10.0.0/16: `snet-aca` 10.10.0.0/23 (delegated to `Microsoft.App/environments`), `snet-pg` 10.10.2.0/24 (delegated to `Microsoft.DBforPostgreSQL/flexibleServers`); private DNS zone `<pg>.private.postgres.database.azure.com` linked to the VNet.
- `pg-kairos-<u>`: PostgreSQL 16, Burstable B1ms, 32 GiB, 7-day backups, no HA, private access only, Entra-only authentication (password authentication disabled, no administrator password anywhere), database `kairos`.
- `cae-kairos-<env>`: workload-profiles environment (Consumption profile) in `snet-aca`, logs to Log Analytics.
- Apps `kairos-extract` and `kairos-predict` (internal ingress, port 8000, `/healthz`), `kairos-demo` (external ingress, port 8501, `/_stcore/health`, Entra built-in authentication once `demoAuthClientId` is set); 0.5 vCPU / 1 GiB, scale 0 to 2.
- Jobs `kairos-job-scenarios`, `kairos-job-train`, `kairos-job-evaluate`: manual trigger, 2 vCPU / 4 GiB, 2-hour timeout, no retries; args `scenarios --all`, `train`, `evaluate --all`.
- `budget-kairos-<env>`: monthly, alerts at 80 % actual and 100 % forecast to `ownerEmail`.

Every app and job receives the variables of `config/app.yaml` (`KAIROS_*`,
`ALLOW_REAL_NOTES_TO_LLM`, `AZURE_CLIENT_ID`, `APPLICATIONINSIGHTS_CONNECTION_STRING`).

## Deploy sequence (Azure CLI path, the reference path)

Prerequisites: Azure CLI 2.87 or later signed in as the owner (`az login`), Python 3.11,
and the Bicep CLI (`az bicep install` runs automatically). Docker is not needed; images
are built by Azure Container Registry.

```powershell
.\scripts\deploy.ps1                       # Windows PowerShell 5.1 or later
```

```bash
bash scripts/deploy.sh --env dev           # Git Bash, Linux, GitHub Actions
```

Both scripts perform the same idempotent sequence and exit non-zero on the first failure:

1. `az account set`; register `Microsoft.App`, `Microsoft.ContainerRegistry`,
   `Microsoft.DBforPostgreSQL`, `Microsoft.KeyVault`, `Microsoft.OperationalInsights`,
   `Microsoft.Storage`, `Microsoft.CognitiveServices`, `Microsoft.ManagedIdentity` and
   wait for `Registered`.
2. `az deployment sub create --location swedencentral --template-file infra/main.bicep
   --parameters infra/main.bicepparam` plus overrides (`environmentName`,
   `resourceGroupName`, `location`, the signed-in user's `deployerObjectId`, and, on a
   re-run, the images and client id already deployed so nothing regresses to the
   placeholder). Outputs are written to `infra/outputs.local.json`.
3. Entra app registration `kairos-demo-<env>` (`AzureADMyOrg`, redirect URI
   `https://<demoFqdn>/.auth/login/aad/callback`, id-token issuance on), its service
   principal with `appRoleAssignmentRequired = true`, the owner (and every id passed with
   `-AllowedUserObjectIds` / `--allowed-user-object-ids`) assigned to the default app
   role, and a client secret created only if the Key Vault secret
   `demo-auth-client-secret` does not exist yet. The secret goes from
   `az ad app credential reset` into a variable and straight into
   `az keyvault secret set`; it is never printed or written to a file.
4. A staging directory is assembled from an allow-list only (`pyproject.toml`,
   `README.md`, `requirements.lock.txt` if present, `.dockerignore`, `src/`, `services/`,
   `config/`, `data/reference/`, `data/derived/aggregates/`,
   `tests/fixtures/synthetic_notes/`), caches removed, then
   `python scripts/privacy_scan.py --root <staging>` must pass before
   `az acr build --registry <acr> --image kairos-<svc>:<tag> --file services/<svc>/Dockerfile
   --build-arg GIT_SHA=<tag> <staging>` runs for `extract`, `predict`, `jobs`, `demo`.
   Tag = short git sha inside a git repository, otherwise a UTC timestamp `yyyyMMddHHmm`.
5. The subscription deployment runs again with the four image references,
   `demoAuthClientId` and `gitSha`, so the apps and jobs pick up the images, probes and
   authentication.
6. The demo URL and the internal URLs are printed.

Switches: `-SkipInfra` / `--skip-infra` (skip step 2, needs `outputs.local.json`; step 5
still runs), `-SkipBuild` / `--skip-build` (skip step 4; `-Tag` / `--tag` selects
existing images, otherwise the deployed ones stay), `-SkipAuth` / `--skip-auth` (skip
step 3; the client id of the deployed auth configuration is re-used).

Expect 12 to 15 minutes for the first run (PostgreSQL and the Container Apps environment
dominate) and a few minutes per image build.

### azd path (optional)

`azure.yaml` describes the same four services for the Azure Developer CLI
(`winget install microsoft.azd`): `azd auth login`, `azd up` provisions `infra/main.bicep`
with `infra/main.bicepparam` and builds the images remotely (`remoteBuild: true`), matching
the apps by their `azd-service-name` tags (`extract`, `predict`, `demo`, and `jobs` on
`kairos-job-scenarios`; the other two jobs receive the same image through step 5 of the
scripts). azd does not create the Entra app registration or the Key Vault secret; run
`scripts/deploy.ps1 -SkipInfra -SkipBuild` once for step 3 and the wiring pass. The
scripts are the path validated here; azd was not installed on the build machine.

## GitHub Actions (OIDC, no stored secrets)

`ci.yml` runs on every push and pull request: `ruff check .`, `pytest -q`,
`python scripts/privacy_scan.py`, `python scripts/export_schemas.py --check`.

`deploy.yml` runs on `workflow_dispatch` (input `environment`, default `dev`) and on
pushes to `main` touching `src/**`, `services/**`, `config/**`, `infra/**` or
`pyproject.toml`. It logs in with `azure/login@v2` using the repository **variables**
`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` and runs
`bash scripts/deploy.sh --env dev --skip-auth`. The CI identity has no Microsoft Graph
permissions, so the app registration is created once by the owner from a workstation;
CI reads the client id back from the deployed auth configuration.

One-time set-up by the owner (the only place an app registration for CI is created):

```bash
SUB=004a53c3-3a7f-4d2f-b288-483265274096
APP_ID=$(az ad app create --display-name kairos-github-deploy --query appId -o tsv)
az ad sp create --id "$APP_ID"
SP_ID=$(az ad sp show --id "$APP_ID" --query id -o tsv)

# Federated credential for the main branch (repeat with subject
# repo:<owner>/<repo>:environment:dev if a GitHub environment is used).
az ad app federated-credential create --id "$APP_ID" --parameters '{
  "name": "kairos-main",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:<owner>/<repo>:ref:refs/heads/main",
  "audiences": ["api://AzureADTokenExchange"]
}'

# Roles: Contributor + User Access Administrator (role assignments in the template),
# either on the subscription or on both resource groups, plus Cognitive Services
# Contributor on the OpenAI resource group (model deployments).
az role assignment create --assignee-object-id "$SP_ID" --assignee-principal-type ServicePrincipal \
  --role Contributor --scope "/subscriptions/$SUB"
az role assignment create --assignee-object-id "$SP_ID" --assignee-principal-type ServicePrincipal \
  --role "User Access Administrator" --scope "/subscriptions/$SUB"
az role assignment create --assignee-object-id "$SP_ID" --assignee-principal-type ServicePrincipal \
  --role "Cognitive Services Contributor" \
  --scope "/subscriptions/$SUB/resourceGroups/rg-papageorgiou.minas-2513"
```

Resource-group scoped alternative: `--scope /subscriptions/$SUB/resourceGroups/rg-kairos-dev`
(create the group first) and the same two roles on `rg-papageorgiou.minas-2513`; the
subscription-level deployment itself then needs at least Reader on the subscription.
Then set the repository variables: `AZURE_CLIENT_ID=$APP_ID`,
`AZURE_TENANT_ID=b38e34be-d9fb-457f-b1b8-d4d3b2ef89fb`, `AZURE_SUBSCRIPTION_ID=$SUB`.

## Teardown

```bash
az group delete --name rg-kairos-dev --yes --no-wait
az cognitiveservices account deployment delete --name papageorgiouminas-0092-resource \
  --resource-group rg-papageorgiou.minas-2513 --deployment-name kairos-extract
az cognitiveservices account deployment delete --name papageorgiouminas-0092-resource \
  --resource-group rg-papageorgiou.minas-2513 --deployment-name kairos-adjudicate
az ad app delete --id "$(az ad app list --display-name kairos-demo-dev --query '[0].appId' -o tsv)"
az keyvault purge --name <kv-kairos-...> --location swedencentral   # optional: frees the soft-deleted name
```

The role assignment on the OpenAI account disappears with the identity. The private DNS
zone, the budget and the Log Analytics workspace live in the resource group and go with
it. Nothing else in the subscription is referenced by the templates.

## Placeholders and open points

- Until the images are built, every app and job runs
  `mcr.microsoft.com/k8se/quickstart:latest`; ingress target ports and health paths are
  those of the real services, so the placeholder revisions report as unhealthy, which is
  expected. HTTP probes are attached only when a real image reference is supplied.
- The demo has no authentication until step 3 has run and the wiring pass has passed
  `demoAuthClientId`; before that the external URL serves the placeholder page.
- `services/*/Dockerfile`, `scripts/privacy_scan.py` and `scripts/export_schemas.py` are
  produced by the application work stream; the deploy scripts fail early when they are
  missing.
- Model deployments are pinned to the given versions with
  `versionUpgradeOption: OnceCurrentVersionExpired`; run
  `az cognitiveservices account list-models` and record the result in `docs/runbook.md`
  before changing them.
- What-if cannot analyse the role assignments and the PostgreSQL administrator whose
  names derive from the identity's object id ("Unsupported"); they are created normally.
