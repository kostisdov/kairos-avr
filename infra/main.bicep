// KAIROS research prototype: subscription-scoped entry point.
//
//   az deployment sub create --location swedencentral \
//     --template-file infra/main.bicep --parameters infra/main.bicepparam
//
// Creates the resource group, the managed identity, the model deployments on the
// owner's EXISTING Azure OpenAI account (never a new account) and everything else
// in resources.bicep. No secrets: managed identity everywhere.
targetScope = 'subscription'

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------
@description('Environment name; part of resource names and the environment tag.')
@minLength(1)
@maxLength(12)
param environmentName string = 'dev'

@description('Azure region for every resource created here.')
param location string = 'swedencentral'

@description('Resource group that receives all KAIROS resources.')
param resourceGroupName string = 'rg-kairos-dev'

@description('Owner e-mail: owner tag and budget notifications. Empty skips the budget.')
param ownerEmail string = ''

@description('Object id of the principal running the deployment; receives Storage Blob Data Contributor and Key Vault Secrets Officer. Empty skips those role assignments.')
param deployerObjectId string = ''

@description('Object id of a user registered as PostgreSQL Entra administrator next to the managed identity. Empty skips.')
param postgresAdminObjectId string = ''

@description('Principal name (UPN) of that user; required when postgresAdminObjectId is set.')
param postgresAdminPrincipalName string = ''

@description('Create the PostgreSQL Entra administrators. Set to false on a re-deployment (e.g. the wire pass) once they already exist, since Postgres rejects re-creating an existing role.')
param deployPostgresAdministrators bool = true

// ---------------------------------------------------------------------------
// Azure OpenAI (existing account, deployments added to it)
// ---------------------------------------------------------------------------
@description('Existing Azure AI Services / Azure OpenAI account name.')
param openAiAccountName string = 'papageorgiouminas-0092-resource'

@description('Resource group of that account.')
param openAiResourceGroup string = 'rg-papageorgiou.minas-2513'

@description('Azure OpenAI API version handed to the services.')
param openAiApiVersion string = '2025-04-01-preview'

param openAiExtractDeploymentName string = 'kairos-extract'
param openAiExtractModelName string = 'gpt-5.1'
param openAiExtractModelVersion string = '2025-11-13'
param openAiExtractSkuName string = 'Standard'
@minValue(1)
param openAiExtractCapacity int = 100

param openAiAdjudicateDeploymentName string = 'kairos-adjudicate'
param openAiAdjudicateModelName string = 'gpt-5.1'
param openAiAdjudicateModelVersion string = '2025-11-13'
param openAiAdjudicateSkuName string = 'Standard'
@minValue(1)
param openAiAdjudicateCapacity int = 50

@description('reasoning_effort sent to the extraction model; empty for a non-reasoning model (temperature 0 is sent instead).')
param openAiExtractReasoningEffort string = 'low'

@description('reasoning_effort sent to the adjudication-assist model.')
param openAiAdjudicateReasoningEffort string = 'medium'

@description('Pre-screen deployment; an empty openAiPrescreenModelName skips it (rules-only pre-screen).')
param openAiPrescreenDeploymentName string = 'kairos-prescreen'
param openAiPrescreenModelName string = ''
param openAiPrescreenModelVersion string = ''
param openAiPrescreenSkuName string = 'Standard'
@minValue(1)
param openAiPrescreenCapacity int = 10

// ---------------------------------------------------------------------------
// Workloads
// ---------------------------------------------------------------------------
@description('Full image references (login-server/repository:tag). Empty means the public placeholder image.')
param extractImage string = ''
param predictImage string = ''
param demoImage string = ''
param jobsImage string = ''

@description('Application (client) id of the kairos-demo-<env> app registration. Empty leaves the demo without built-in authentication.')
param demoAuthClientId string = ''

@description('Single switch permitting de-identified real note text to be sent to Azure OpenAI.')
param allowRealNotesToLlm bool = true

@description('Monthly Cost Management budget on the resource group.')
@minValue(1)
param budgetAmount int = 150

@description('Budget start date (first of a month, yyyy-MM-dd).')
param budgetStartDate string = utcNow('yyyy-MM-01')

@description('Git commit baked into the images; exposed as KAIROS_GIT_SHA.')
param gitSha string = ''

// ---------------------------------------------------------------------------
var tags = {
  project: 'kairos'
  owner: ownerEmail
  environment: environmentName
  'azd-env-name': environmentName
}

resource rg 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: resourceGroupName
  location: location
  tags: tags
}

module identity 'identity.bicep' = {
  name: 'kairos-identity'
  scope: rg
  params: {
    name: 'id-kairos-${environmentName}'
    location: location
    tags: tags
  }
}

module openai 'openai.bicep' = {
  name: 'kairos-openai'
  scope: resourceGroup(openAiResourceGroup)
  params: {
    accountName: openAiAccountName
    identityPrincipalId: identity.outputs.principalId
    extractDeploymentName: openAiExtractDeploymentName
    extractModelName: openAiExtractModelName
    extractModelVersion: openAiExtractModelVersion
    extractSkuName: openAiExtractSkuName
    extractCapacity: openAiExtractCapacity
    adjudicateDeploymentName: openAiAdjudicateDeploymentName
    adjudicateModelName: openAiAdjudicateModelName
    adjudicateModelVersion: openAiAdjudicateModelVersion
    adjudicateSkuName: openAiAdjudicateSkuName
    adjudicateCapacity: openAiAdjudicateCapacity
    prescreenDeploymentName: openAiPrescreenDeploymentName
    prescreenModelName: openAiPrescreenModelName
    prescreenModelVersion: openAiPrescreenModelVersion
    prescreenSkuName: openAiPrescreenSkuName
    prescreenCapacity: openAiPrescreenCapacity
  }
}

module resources 'resources.bicep' = {
  name: 'kairos-resources'
  scope: rg
  params: {
    environmentName: environmentName
    location: location
    tags: tags
    identityName: identity.outputs.name
    identityPrincipalId: identity.outputs.principalId
    identityClientId: identity.outputs.clientId
    deployerObjectId: deployerObjectId
    postgresAdminObjectId: postgresAdminObjectId
    postgresAdminPrincipalName: postgresAdminPrincipalName
    deployPostgresAdministrators: deployPostgresAdministrators
    openAiEndpoint: openai.outputs.endpoint
    openAiApiVersion: openAiApiVersion
    openAiExtractDeploymentName: openai.outputs.extractDeploymentName
    openAiPrescreenDeploymentName: openai.outputs.prescreenDeploymentName
    openAiAdjudicateDeploymentName: openai.outputs.adjudicateDeploymentName
    openAiExtractReasoningEffort: openAiExtractReasoningEffort
    openAiAdjudicateReasoningEffort: openAiAdjudicateReasoningEffort
    extractImage: extractImage
    predictImage: predictImage
    demoImage: demoImage
    jobsImage: jobsImage
    demoAuthClientId: demoAuthClientId
    allowRealNotesToLlm: allowRealNotesToLlm
    ownerEmail: ownerEmail
    budgetAmount: budgetAmount
    budgetStartDate: budgetStartDate
    gitSha: gitSha
  }
}

// ---------------------------------------------------------------------------
// Outputs (scripts/deploy.* store them in infra/outputs.local.json)
// ---------------------------------------------------------------------------
output resourceGroupName string = rg.name
output identityName string = identity.outputs.name
output identityClientId string = identity.outputs.clientId
output identityPrincipalId string = identity.outputs.principalId
output acrName string = resources.outputs.acrName
output acrLoginServer string = resources.outputs.acrLoginServer
output storageAccountName string = resources.outputs.storageAccountName
output storageAccountUrl string = resources.outputs.storageAccountUrl
output keyVaultName string = resources.outputs.keyVaultName
output keyVaultUrl string = resources.outputs.keyVaultUrl
output postgresFqdn string = resources.outputs.postgresFqdn
output postgresServerName string = resources.outputs.postgresServerName
output postgresDatabaseName string = resources.outputs.postgresDatabaseName
output containerAppsEnvironmentName string = resources.outputs.containerAppsEnvironmentName
output environmentDefaultDomain string = resources.outputs.environmentDefaultDomain
output demoFqdn string = resources.outputs.demoFqdn
output extractInternalUrl string = resources.outputs.extractInternalUrl
output predictInternalUrl string = resources.outputs.predictInternalUrl
output openAiEndpoint string = openai.outputs.endpoint
output appInsightsConnectionString string = resources.outputs.appInsightsConnectionString
// Effective values, so that a re-run of the deploy scripts can preserve them.
output extractImage string = resources.outputs.extractImage
output predictImage string = resources.outputs.predictImage
output demoImage string = resources.outputs.demoImage
output jobsImage string = resources.outputs.jobsImage
output demoAuthClientId string = resources.outputs.demoAuthClientId
