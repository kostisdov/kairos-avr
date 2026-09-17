// KAIROS: model deployments on the owner's EXISTING Azure AI Services (Azure OpenAI)
// account plus the "Cognitive Services OpenAI User" role for the managed identity.
// Deployed at the scope of the resource group that holds the account. The account
// itself and any deployment not named here are never touched.
targetScope = 'resourceGroup'

@description('Name of the existing Azure AI Services / Azure OpenAI account in this resource group.')
param accountName string

@description('Principal id (object id) of the user-assigned managed identity that calls the models.')
param identityPrincipalId string

@description('Extraction deployment (structured output).')
param extractDeploymentName string = 'kairos-extract'
param extractModelName string = 'gpt-5.1'
param extractModelVersion string = '2025-11-13'
param extractSkuName string = 'Standard'
@minValue(1)
param extractCapacity int = 100

@description('Adjudication-assist deployment (reasoning model).')
param adjudicateDeploymentName string = 'kairos-adjudicate'
param adjudicateModelName string = 'gpt-5.1'
param adjudicateModelVersion string = '2025-11-13'
param adjudicateSkuName string = 'Standard'
@minValue(1)
param adjudicateCapacity int = 50

@description('Pre-screen deployment. An empty prescreenModelName skips it (rules-only pre-screen).')
param prescreenDeploymentName string = 'kairos-prescreen'
param prescreenModelName string = ''
param prescreenModelVersion string = ''
param prescreenSkuName string = 'Standard'
@minValue(1)
param prescreenCapacity int = 10

// Cognitive Services OpenAI User
var openAiUserRoleId = '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'

resource account 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = {
  name: accountName
}

var requiredDeployments = [
  {
    name: extractDeploymentName
    model: extractModelName
    version: extractModelVersion
    sku: extractSkuName
    capacity: extractCapacity
  }
  {
    name: adjudicateDeploymentName
    model: adjudicateModelName
    version: adjudicateModelVersion
    sku: adjudicateSkuName
    capacity: adjudicateCapacity
  }
]

var optionalDeployments = empty(prescreenModelName) ? [] : [
  {
    name: prescreenDeploymentName
    model: prescreenModelName
    version: prescreenModelVersion
    sku: prescreenSkuName
    capacity: prescreenCapacity
  }
]

var deploymentList = concat(requiredDeployments, optionalDeployments)

// Model deployments on one account must be created one at a time.
@batchSize(1)
resource deployments 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = [for d in deploymentList: {
  parent: account
  name: d.name
  sku: {
    name: d.sku
    capacity: d.capacity
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: d.model
      version: d.version
    }
    // Pinned for reproducibility; upgraded only when the pinned version retires.
    versionUpgradeOption: 'OnceCurrentVersionExpired'
  }
}]

resource openAiUserRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(account.id, identityPrincipalId, openAiUserRoleId)
  scope: account
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', openAiUserRoleId)
    principalId: identityPrincipalId
    principalType: 'ServicePrincipal'
  }
}

// The Azure OpenAI endpoint is https://<custom subdomain>.openai.azure.com/ ; the
// custom subdomain normally equals the account name.
var customSubDomain = account.properties.?customSubDomainName ?? accountName

output accountId string = account.id
output endpoint string = 'https://${customSubDomain}.openai.azure.com/'
output extractDeploymentName string = extractDeploymentName
output adjudicateDeploymentName string = adjudicateDeploymentName
output prescreenDeploymentName string = empty(prescreenModelName) ? '' : prescreenDeploymentName
