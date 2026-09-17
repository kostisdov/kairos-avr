#!/usr/bin/env bash
# KAIROS: provision the Azure infrastructure, register the demo Entra app, build the
# container images in Azure Container Registry and wire them into the Container Apps.
#
#   bash scripts/deploy.sh [--env dev] [--subscription <id>] [--location swedencentral]
#                          [--resource-group rg-kairos-<env>] [--tag <image tag>]
#                          [--owner-object-id <id>] [--allowed-user-object-ids id1,id2]
#                          [--skip-infra] [--skip-build] [--skip-auth]
#
# Sequence (idempotent; every step may be re-run):
#   1. az account set; register the resource providers and wait for "Registered"
#   2. az deployment sub create with infra/main.bicep + infra/main.bicepparam
#      (outputs -> infra/outputs.local.json, git-ignored)
#   3. Entra app registration kairos-demo-<env> for the demo login, assignment
#      required, owner (+ optional users) assigned, client secret stored ONCE in
#      Key Vault (never printed, never written to a file)
#   4. az acr build of extract, predict, jobs and demo from an allow-listed staging
#      directory that is privacy-scanned before anything is uploaded
#   5. second deployment passing the image references and the demo client id
#   6. print the demo URL and the internal URLs
#
# Flags: --skip-infra skips step 2 (needs infra/outputs.local.json from an earlier
# run; the final pass still runs), --skip-build skips step 4 (re-uses --tag or the
# images currently deployed), --skip-auth skips step 3 (re-uses the client id of the
# deployed auth configuration, if any). Never enable `set -x`: step 3 handles a secret.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Defaults and constants
# ---------------------------------------------------------------------------
ENV_NAME="dev"
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-004a53c3-3a7f-4d2f-b288-483265274096}"
LOCATION="swedencentral"
RESOURCE_GROUP=""
IMAGE_TAG=""
OWNER_OBJECT_ID="31b7b1d3-2262-4032-8af2-6df67ea834b4"
ALLOWED_USER_OBJECT_IDS=""
SKIP_INFRA=0
SKIP_BUILD=0
SKIP_AUTH=0

TEMPLATE_FILE="infra/main.bicep"
PARAM_FILE="infra/main.bicepparam"
OUTPUTS_FILE="infra/outputs.local.json"
PLACEHOLDER_IMAGE="mcr.microsoft.com/k8se/quickstart:latest"
DEMO_SECRET_NAME="demo-auth-client-secret"
DEFAULT_APP_ROLE_ID="00000000-0000-0000-0000-000000000000"
GRAPH="https://graph.microsoft.com/v1.0"
ARM="https://management.azure.com"
SERVICES=(extract predict jobs demo)
PROVIDERS=(
  Microsoft.App
  Microsoft.ContainerRegistry
  Microsoft.DBforPostgreSQL
  Microsoft.KeyVault
  Microsoft.OperationalInsights
  Microsoft.Storage
  Microsoft.CognitiveServices
  Microsoft.ManagedIdentity
)

APP_ID=""
STAGING_DIR=""

usage() {
  sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

log()  { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)                      [[ $# -ge 2 ]] || die "--env needs a value"; ENV_NAME="$2"; shift 2 ;;
    --subscription)             [[ $# -ge 2 ]] || die "--subscription needs a value"; SUBSCRIPTION_ID="$2"; shift 2 ;;
    --location)                 [[ $# -ge 2 ]] || die "--location needs a value"; LOCATION="$2"; shift 2 ;;
    --resource-group)           [[ $# -ge 2 ]] || die "--resource-group needs a value"; RESOURCE_GROUP="$2"; shift 2 ;;
    --tag)                      [[ $# -ge 2 ]] || die "--tag needs a value"; IMAGE_TAG="$2"; shift 2 ;;
    --owner-object-id)          [[ $# -ge 2 ]] || die "--owner-object-id needs a value"; OWNER_OBJECT_ID="$2"; shift 2 ;;
    --allowed-user-object-ids)  [[ $# -ge 2 ]] || die "--allowed-user-object-ids needs a value"; ALLOWED_USER_OBJECT_IDS="$2"; shift 2 ;;
    --skip-infra)               SKIP_INFRA=1; shift ;;
    --skip-build)               SKIP_BUILD=1; shift ;;
    --skip-auth)                SKIP_AUTH=1; shift ;;
    -h|--help)                  usage; exit 0 ;;
    *)                          usage; die "unknown argument: $1" ;;
  esac
done
if [[ -z "${RESOURCE_GROUP}" ]]; then
  RESOURCE_GROUP="rg-kairos-${ENV_NAME}"
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PYTHON=""
for candidate in python python3; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    PYTHON="${candidate}"
    break
  fi
done
[[ -n "${PYTHON}" ]] || die "python is required (privacy scan and output parsing)"
command -v az >/dev/null 2>&1 || die "Azure CLI (az) is required"

# Windows Git Bash: hand native programs a Windows path.
native_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -w "$1"
  else
    printf '%s' "$1"
  fi
}

cleanup() {
  if [[ -n "${STAGING_DIR}" && -d "${STAGING_DIR}" ]]; then
    rm -rf "${STAGING_DIR}"
  fi
}
trap cleanup EXIT

read_output() {
  [[ -f "${OUTPUTS_FILE}" ]] || die "${OUTPUTS_FILE} not found; run once without --skip-infra"
  "${PYTHON}" - "${OUTPUTS_FILE}" "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    outputs = json.load(fh)
entry = outputs.get(sys.argv[2]) or {}
print(entry.get("value", "") or "")
PY
}

# add_param <array name> <parameter> <value>: appends "parameter=value" when value is non-empty.
add_param() {
  local -n target="$1"
  if [[ -n "$3" ]]; then
    target+=("$2=$3")
  fi
}

run_deployment() {
  local label="$1"
  shift
  local name
  name="kairos-${ENV_NAME}-${label}-$(date -u +%Y%m%d%H%M%S)"
  log "Deployment ${name} (${TEMPLATE_FILE} + ${PARAM_FILE})"
  if [[ $# -gt 0 ]]; then
    info "overrides: $*"
  fi
  az deployment sub create \
    --name "${name}" \
    --location "${LOCATION}" \
    --template-file "${TEMPLATE_FILE}" \
    --parameters "${PARAM_FILE}" \
    --parameters "environmentName=${ENV_NAME}" "resourceGroupName=${RESOURCE_GROUP}" "location=${LOCATION}" "$@" \
    --query properties.outputs \
    --output json > "${OUTPUTS_FILE}.tmp"
  mv -f "${OUTPUTS_FILE}.tmp" "${OUTPUTS_FILE}"
  info "outputs written to ${OUTPUTS_FILE}"
}

register_providers() {
  log "Registering resource providers"
  local ns state attempt
  for ns in "${PROVIDERS[@]}"; do
    state="$(az provider show --namespace "${ns}" --query registrationState -o tsv 2>/dev/null || echo Unknown)"
    if [[ "${state}" != "Registered" ]]; then
      az provider register --namespace "${ns}" --output none
    fi
  done
  for ns in "${PROVIDERS[@]}"; do
    state=""
    for attempt in $(seq 1 60); do
      state="$(az provider show --namespace "${ns}" --query registrationState -o tsv)"
      if [[ "${state}" == "Registered" ]]; then
        break
      fi
      info "${ns}: ${state} (waiting, attempt ${attempt})"
      sleep 10
    done
    [[ "${state}" == "Registered" ]] || die "provider ${ns} did not reach Registered"
    info "${ns}: Registered"
  done
}

# deployerObjectId override: the signed-in user; service principals keep the
# bicepparam default (they hold ARM roles only, see infra/README.md).
resolve_deployer_override() {
  local account_type oid
  account_type="$(az account show --query user.type -o tsv 2>/dev/null || true)"
  if [[ "${account_type}" == "user" ]]; then
    oid="$(az ad signed-in-user show --query id -o tsv 2>/dev/null || true)"
    if [[ -n "${oid}" ]]; then
      printf 'deployerObjectId=%s' "${oid}"
    fi
  fi
}

rg_exists() {
  [[ "$(az group exists --name "${RESOURCE_GROUP}" -o tsv 2>/dev/null || echo false)" == "true" ]]
}

# current_image <resource type> <name> <newline-separated existing names>
current_image() {
  local image
  if ! grep -qx "$2" <<< "$3"; then
    printf ''
    return
  fi
  image="$(az resource show --resource-group "${RESOURCE_GROUP}" --resource-type "$1" --name "$2" \
    --api-version 2024-03-01 --query 'properties.template.containers[0].image' -o tsv 2>/dev/null || true)"
  if [[ "${image}" == "${PLACEHOLDER_IMAGE}" ]]; then
    image=""
  fi
  printf '%s' "${image}"
}

current_auth_client_id() {
  az rest --method get \
    --url "${ARM}/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP}/providers/Microsoft.App/containerApps/kairos-demo/authConfigs?api-version=2024-03-01" \
    --query "value[?name=='current'].properties.identityProviders.azureActiveDirectory.registration.clientId" \
    -o tsv 2>/dev/null | head -n 1 || true
}

# ---------------------------------------------------------------------------
# Step 3: Entra app registration for the demo login
# ---------------------------------------------------------------------------
graph_call() {  # graph_call <method> <url> [json body]
  local method="$1" url="$2" body="${3:-}" tmp rc
  if [[ -n "${body}" ]]; then
    tmp="$(mktemp)"
    printf '%s' "${body}" > "${tmp}"
    az rest --method "${method}" --url "${url}" --headers "Content-Type=application/json" \
      --body "@$(native_path "${tmp}")" --output none
    rc=$?
    rm -f "${tmp}"
    return "${rc}"
  fi
  az rest --method "${method}" --url "${url}" --output none
}

principal_assigned() {  # principal_assigned <sp id> <principal id>
  local assigned
  assigned="$(az rest --method get --url "${GRAPH}/servicePrincipals/$1/appRoleAssignedTo" --query "value[].principalId" -o tsv)"
  grep -qix "$2" <<< "${assigned}"
}

kv_secret_state() {  # 0 exists, 1 missing, 2 error (for example RBAC not propagated yet)
  local out
  if out="$(az keyvault secret show --vault-name "$1" --name "$2" --query id -o tsv 2>&1)"; then
    return 0
  fi
  if grep -qiE "SecretNotFound|was not found" <<< "${out}"; then
    return 1
  fi
  printf '%s\n' "${out}" >&2
  return 2
}

ensure_demo_client_secret() {  # ensure_demo_client_secret <key vault name>
  local kv_name="$1" attempt state=2 secret_value
  for attempt in 1 2 3 4 5 6 7 8; do
    set +e
    kv_secret_state "${kv_name}" "${DEMO_SECRET_NAME}"
    state=$?
    set -e
    if [[ "${state}" -eq 0 ]]; then
      info "Key Vault secret ${DEMO_SECRET_NAME} exists; not rotating"
      return 0
    fi
    if [[ "${state}" -eq 1 ]]; then
      break
    fi
    info "Key Vault not readable yet (role assignment propagating?); retry ${attempt} in 20 s"
    sleep 20
  done
  [[ "${state}" -eq 1 ]] || die "cannot read Key Vault ${kv_name}"
  info "creating a client secret for ${APP_ID} and storing it in ${kv_name}/${DEMO_SECRET_NAME}"
  secret_value="$(az ad app credential reset --id "${APP_ID}" --append --display-name kairos-demo --years 1 --query password -o tsv)"
  [[ -n "${secret_value}" ]] || die "credential reset returned no password"
  az keyvault secret set --vault-name "${kv_name}" --name "${DEMO_SECRET_NAME}" --value "${secret_value}" --query id -o tsv >/dev/null
  unset secret_value
  info "stored"
}

ensure_demo_app_registration() {  # ensure_demo_app_registration <demo fqdn> <key vault name>
  local demo_fqdn="$1" kv_name="$2"
  local app_name="kairos-demo-${ENV_NAME}"
  local redirect_uri="https://${demo_fqdn}/.auth/login/aad/callback"
  log "Entra app registration ${app_name}"
  [[ -n "${demo_fqdn}" ]] || die "demo FQDN unknown; cannot set the redirect URI"

  APP_ID="$(az ad app list --display-name "${app_name}" --query "[0].appId" -o tsv)"
  if [[ -z "${APP_ID}" ]]; then
    info "creating ${app_name}"
    APP_ID="$(az ad app create --display-name "${app_name}" --sign-in-audience AzureADMyOrg \
      --web-redirect-uris "${redirect_uri}" --enable-id-token-issuance true --query appId -o tsv)"
  else
    info "found ${app_name} (${APP_ID}); refreshing the redirect URI"
    az ad app update --id "${APP_ID}" --web-redirect-uris "${redirect_uri}" --enable-id-token-issuance true --output none
  fi
  [[ -n "${APP_ID}" ]] || die "could not determine the application id of ${app_name}"

  local sp_id="" attempt
  for attempt in 1 2 3 4 5 6; do
    sp_id="$(az ad sp list --filter "appId eq '${APP_ID}'" --query "[0].id" -o tsv)"
    if [[ -n "${sp_id}" ]]; then
      break
    fi
    sp_id="$(az ad sp create --id "${APP_ID}" --query id -o tsv 2>/dev/null || true)"
    if [[ -n "${sp_id}" ]]; then
      break
    fi
    info "service principal not ready yet (attempt ${attempt}); waiting 10 s"
    sleep 10
  done
  [[ -n "${sp_id}" ]] || die "could not create the service principal for ${APP_ID}"

  info "requiring assignment on the service principal ${sp_id}"
  graph_call PATCH "${GRAPH}/servicePrincipals/${sp_id}" '{"appRoleAssignmentRequired": true}'

  local principals=("${OWNER_OBJECT_ID}") extra principal
  if [[ -n "${ALLOWED_USER_OBJECT_IDS}" ]]; then
    IFS=',' read -r -a extra <<< "${ALLOWED_USER_OBJECT_IDS}"
    principals+=("${extra[@]}")
  fi
  for principal in "${principals[@]}"; do
    principal="$(printf '%s' "${principal}" | tr -d '[:space:]')"
    [[ -n "${principal}" ]] || continue
    if principal_assigned "${sp_id}" "${principal}"; then
      info "assignment exists: ${principal}"
      continue
    fi
    if graph_call POST "${GRAPH}/servicePrincipals/${sp_id}/appRoleAssignedTo" \
        "{\"principalId\":\"${principal}\",\"resourceId\":\"${sp_id}\",\"appRoleId\":\"${DEFAULT_APP_ROLE_ID}\"}"; then
      info "assigned ${principal}"
    elif principal_assigned "${sp_id}" "${principal}"; then
      # Graph answers 400 "Permission being assigned already exists" for duplicates.
      info "assignment exists: ${principal}"
    else
      die "could not assign ${principal} to ${app_name}"
    fi
  done

  ensure_demo_client_secret "${kv_name}"
}

# ---------------------------------------------------------------------------
# Step 4: build context and images
# ---------------------------------------------------------------------------
copy_into_staging() {
  local item="$1" parent
  parent="$(dirname "${item}")"
  mkdir -p "${STAGING_DIR}/${parent}"
  cp -R "${item}" "${STAGING_DIR}/${parent}/"
}

stage_build_context() {
  STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kairos-build.XXXXXX")"
  log "Staging the build context in ${STAGING_DIR} (allow-list only)"
  local required=(pyproject.toml src services config)
  local optional=(README.md requirements.lock.txt .dockerignore data/reference data/derived/aggregates tests/fixtures/synthetic_notes)
  local item
  for item in "${required[@]}"; do
    [[ -e "${item}" ]] || die "required build input missing: ${item}"
    copy_into_staging "${item}"
  done
  for item in "${optional[@]}"; do
    if [[ -e "${item}" ]]; then
      copy_into_staging "${item}"
    else
      info "optional input absent, skipped: ${item}"
    fi
  done
  # Never ship caches or build metadata.
  find "${STAGING_DIR}" \( -name __pycache__ -o -name '*.egg-info' -o -name .pytest_cache -o -name .ruff_cache \) -type d -prune -exec rm -rf {} +
  find "${STAGING_DIR}" -name '*.pyc' -type f -delete
  # The allow-list cannot contain these; check anyway before anything leaves the machine.
  if (cd "${STAGING_DIR}" && find . \( -iname '*.xlsx' -o -path './data/raw*' -o -path './data/derived/private*' -o -path './tmp*' \) | grep -q .); then
    die "forbidden material found in the staging directory; nothing was uploaded"
  fi
}

run_privacy_scan() {
  log "Privacy scan of the staging directory"
  [[ -f scripts/privacy_scan.py ]] || die "scripts/privacy_scan.py is missing"
  "${PYTHON}" scripts/privacy_scan.py --root "$(native_path "${STAGING_DIR}")" || die "privacy scan failed; nothing was uploaded"
}

resolve_image_tag() {
  if [[ -n "${IMAGE_TAG}" ]]; then
    return
  fi
  local sha=""
  if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    sha="$(git rev-parse --short HEAD 2>/dev/null || true)"
  fi
  if [[ -n "${sha}" ]]; then
    IMAGE_TAG="${sha}"
  else
    IMAGE_TAG="$(date -u +%Y%m%d%H%M)"
  fi
}

build_images() {  # build_images <acr name>
  local acr_name="$1" svc
  for svc in "${SERVICES[@]}"; do
    [[ -f "${STAGING_DIR}/services/${svc}/Dockerfile" ]] || die "services/${svc}/Dockerfile is missing"
  done
  for svc in "${SERVICES[@]}"; do
    log "az acr build kairos-${svc}:${IMAGE_TAG}"
    az acr build --registry "${acr_name}" --image "kairos-${svc}:${IMAGE_TAG}" \
      --file "services/${svc}/Dockerfile" --build-arg "GIT_SHA=${IMAGE_TAG}" "$(native_path "${STAGING_DIR}")"
  done
}

# ---------------------------------------------------------------------------
# Main sequence
# ---------------------------------------------------------------------------
log "KAIROS deploy: env=${ENV_NAME} resource-group=${RESOURCE_GROUP} location=${LOCATION} subscription=${SUBSCRIPTION_ID}"

# 1. Subscription and providers
az account set --subscription "${SUBSCRIPTION_ID}"
register_providers

# Current state, so that a re-run never resets images or auth to placeholders.
extract_image=""
predict_image=""
demo_image=""
jobs_image=""
auth_client_id=""
if rg_exists; then
  existing_apps="$(az resource list --resource-group "${RESOURCE_GROUP}" --resource-type Microsoft.App/containerApps --query '[].name' -o tsv 2>/dev/null || true)"
  existing_jobs="$(az resource list --resource-group "${RESOURCE_GROUP}" --resource-type Microsoft.App/jobs --query '[].name' -o tsv 2>/dev/null || true)"
  extract_image="$(current_image Microsoft.App/containerApps kairos-extract "${existing_apps}")"
  predict_image="$(current_image Microsoft.App/containerApps kairos-predict "${existing_apps}")"
  demo_image="$(current_image Microsoft.App/containerApps kairos-demo "${existing_apps}")"
  jobs_image="$(current_image Microsoft.App/jobs kairos-job-scenarios "${existing_jobs}")"
  if grep -qx "kairos-demo" <<< "${existing_apps}"; then
    auth_client_id="$(current_auth_client_id)"
  fi
fi

common_params=()
deployer_override="$(resolve_deployer_override)"
if [[ -n "${deployer_override}" ]]; then
  common_params+=("${deployer_override}")
fi

# 2. Provisioning pass
if [[ "${SKIP_INFRA}" -eq 1 ]]; then
  log "Skipping the provisioning pass (--skip-infra)"
  [[ -f "${OUTPUTS_FILE}" ]] || die "${OUTPUTS_FILE} not found; run once without --skip-infra"
else
  provision_params=(${common_params[@]+"${common_params[@]}"})
  add_param provision_params extractImage "${extract_image}"
  add_param provision_params predictImage "${predict_image}"
  add_param provision_params demoImage "${demo_image}"
  add_param provision_params jobsImage "${jobs_image}"
  add_param provision_params demoAuthClientId "${auth_client_id}"
  if [[ -n "${extract_image}" ]]; then
    add_param provision_params gitSha "${extract_image##*:}"
  fi
  run_deployment provision ${provision_params[@]+"${provision_params[@]}"}
fi

acr_name="$(read_output acrName)"
acr_login_server="$(read_output acrLoginServer)"
key_vault_name="$(read_output keyVaultName)"
demo_fqdn="$(read_output demoFqdn)"
[[ -n "${acr_name}" && -n "${acr_login_server}" && -n "${key_vault_name}" ]] || die "deployment outputs incomplete in ${OUTPUTS_FILE}"

# 3. Demo authentication
if [[ "${SKIP_AUTH}" -eq 1 ]]; then
  log "Skipping the Entra app registration (--skip-auth)"
  APP_ID="${auth_client_id}"
  if [[ -n "${APP_ID}" ]]; then
    info "re-using the client id of the deployed auth configuration: ${APP_ID}"
  else
    info "no auth configuration deployed yet; the demo stays without built-in authentication"
  fi
else
  ensure_demo_app_registration "${demo_fqdn}" "${key_vault_name}"
fi

# 4. Images
if [[ "${SKIP_BUILD}" -eq 1 ]]; then
  log "Skipping the image build (--skip-build)"
  if [[ -n "${IMAGE_TAG}" ]]; then
    for svc in "${SERVICES[@]}"; do
      declare "${svc}_image=${acr_login_server}/kairos-${svc}:${IMAGE_TAG}"
    done
    info "re-using tag ${IMAGE_TAG}"
  else
    info "keeping the images currently deployed"
  fi
else
  resolve_image_tag
  stage_build_context
  run_privacy_scan
  build_images "${acr_name}"
  for svc in "${SERVICES[@]}"; do
    declare "${svc}_image=${acr_login_server}/kairos-${svc}:${IMAGE_TAG}"
  done
fi

# 5. Wire images and authentication in
final_params=(${common_params[@]+"${common_params[@]}"})
add_param final_params extractImage "${extract_image}"
add_param final_params predictImage "${predict_image}"
add_param final_params demoImage "${demo_image}"
add_param final_params jobsImage "${jobs_image}"
add_param final_params demoAuthClientId "${APP_ID}"
if [[ -n "${IMAGE_TAG}" ]]; then
  add_param final_params gitSha "${IMAGE_TAG}"
elif [[ -n "${extract_image}" ]]; then
  add_param final_params gitSha "${extract_image##*:}"
fi
if [[ -n "${extract_image}${predict_image}${demo_image}${jobs_image}${APP_ID}" ]]; then
  run_deployment wire ${final_params[@]+"${final_params[@]}"}
else
  log "No images or authentication to wire in; the apps keep the placeholder image"
fi

# 6. Summary
demo_fqdn="$(read_output demoFqdn)"
log "Done"
info "demo URL:            https://${demo_fqdn}"
info "extract (internal):  $(read_output extractInternalUrl)"
info "predict (internal):  $(read_output predictInternalUrl)"
info "registry:            ${acr_login_server}"
info "key vault:           ${key_vault_name}"
info "postgres:            $(read_output postgresFqdn)"
if [[ -n "${APP_ID}" ]]; then
  info "demo auth client id: ${APP_ID}"
else
  info "demo auth:           not configured (run without --skip-auth)"
fi
info "outputs:             ${OUTPUTS_FILE}"
