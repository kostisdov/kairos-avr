// KAIROS: every resource in rg-kairos-<env> except the managed identity (identity.bicep)
// and the model deployments on the existing Azure OpenAI account (openai.bicep).
// See docs/kairos_azure_build_design.md section 6 and config/app.yaml.
targetScope = 'resourceGroup'

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
@description('Environment name (dev). Part of resource names and the KAIROS_ENV variable.')
param environmentName string

@description('Azure region.')
param location string = resourceGroup().location

@description('Tags applied to every resource.')
param tags object = {}

@description('Name of the user-assigned managed identity created by identity.bicep.')
param identityName string

@description('Principal id (object id) of that identity.')
param identityPrincipalId string

@description('Client id of that identity (AZURE_CLIENT_ID for DefaultAzureCredential).')
param identityClientId string

@description('Object id of the deploying principal; receives Storage Blob Data Contributor and Key Vault Secrets Officer. Empty skips.')
param deployerObjectId string = ''

@description('Object id of a user to register as PostgreSQL Entra administrator. Empty skips.')
param postgresAdminObjectId string = ''

@description('Principal name (UPN) of that user. Required when postgresAdminObjectId is set.')
param postgresAdminPrincipalName string = ''

@description('Create the PostgreSQL Entra administrators. The Postgres engine executes CREATE ROLE on every deployment of these resources, which fails with a duplicate-role error if they already exist; set to false once they have been created so later re-deployments (e.g. the wire pass) do not redeclare them.')
param deployPostgresAdministrators bool = true

@description('Azure OpenAI endpoint, https://<account>.openai.azure.com/')
param openAiEndpoint string

@description('Azure OpenAI API version used by the services.')
param openAiApiVersion string = '2025-04-01-preview'

@description('Extraction deployment name handed to the services.')
param openAiExtractDeploymentName string = 'kairos-extract'

@description('Pre-screen deployment name handed to the services; empty means rules-only pre-screen.')
param openAiPrescreenDeploymentName string = ''

@description('Adjudication deployment name handed to the services.')
param openAiAdjudicateDeploymentName string = 'kairos-adjudicate'

@description('reasoning_effort for the extraction model; empty for a non-reasoning model.')
param openAiExtractReasoningEffort string = 'low'

@description('reasoning_effort for the adjudication-assist model.')
param openAiAdjudicateReasoningEffort string = 'medium'

@description('Container image reference for kairos-extract. Empty means the public placeholder image until the real image is built.')
param extractImage string = ''

@description('Container image reference for kairos-predict.')
param predictImage string = ''

@description('Container image reference for kairos-demo.')
param demoImage string = ''

@description('Container image reference for the three jobs.')
param jobsImage string = ''

@description('Application (client) id of the Entra app registration protecting the demo. Empty disables built-in authentication.')
param demoAuthClientId string = ''

@description('Single switch permitting de-identified real note text to be sent to Azure OpenAI (design rule 0.3).')
param allowRealNotesToLlm bool = true

@description('Owner e-mail for budget notifications. Empty skips the budget.')
param ownerEmail string = ''

@description('Monthly budget in the billing currency.')
@minValue(1)
param budgetAmount int = 150

@description('Budget start date, first of a month, yyyy-MM-dd.')
param budgetStartDate string

@description('Git commit baked into the images; exposed as KAIROS_GIT_SHA.')
param gitSha string = ''

@description('Blob containers created in the storage account.')
param blobContainerNames array = [
  'scenarios'
  'models'
  'metrics'
  'figures'
  'reference'
  'aggregates'
  'raw'
  'calibration'
]

// ---------------------------------------------------------------------------
// Names and constants
// ---------------------------------------------------------------------------
var unique = uniqueString(resourceGroup().id)
var placeholderImage = 'mcr.microsoft.com/k8se/quickstart:latest'

var logAnalyticsName = 'log-kairos-${environmentName}'
var appInsightsName = 'appi-kairos-${environmentName}'
var acrName = 'crkairos${unique}'
var storageAccountName = 'stkairos${unique}'
var keyVaultName = 'kv-kairos-${unique}' // 23 characters
var vnetName = 'vnet-kairos-${environmentName}'
var pgServerName = 'pg-kairos-${unique}'
var pgDatabaseName = 'kairos'
var pgDnsZoneName = '${pgServerName}.private.postgres.database.azure.com'
var containerAppsEnvName = 'cae-kairos-${environmentName}'
var budgetName = 'budget-kairos-${environmentName}'
var demoAuthSecretName = 'demo-auth-client-secret'

var roleIds = {
  acrPull: '7f951dda-4ed3-4680-a7ca-43fe172d538d'
  storageBlobDataContributor: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
  keyVaultSecretsUser: '4633458b-17de-408a-b874-0445c86b69e6'
  keyVaultSecretsOfficer: 'b86a8fe4-44ce-4948-aee5-eccb2c155cd7'
}

var storageAccountUrl = 'https://${storageAccountName}.blob.${environment().suffixes.storage}'
var keyVaultUrl = 'https://${keyVaultName}${environment().suffixes.keyvaultDns}'
var demoAuthSecretUrl = '${keyVaultUrl}/secrets/${demoAuthSecretName}'
var demoAuthEnabled = !empty(demoAuthClientId)

var extractImageEffective = empty(extractImage) ? placeholderImage : extractImage
var predictImageEffective = empty(predictImage) ? placeholderImage : predictImage
var demoImageEffective = empty(demoImage) ? placeholderImage : demoImage
var jobsImageEffective = empty(jobsImage) ? placeholderImage : jobsImage

// ---------------------------------------------------------------------------
// Existing identity (created by identity.bicep in the same resource group)
// ---------------------------------------------------------------------------
resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: identityName
}

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
    workspaceCapping: {
      dailyQuotaGb: 1
    }
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
    IngestionMode: 'LogAnalytics'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// ---------------------------------------------------------------------------
// Container registry
// ---------------------------------------------------------------------------
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  tags: tags
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled'
  }
}

resource acrPullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, identityPrincipalId, roleIds.acrPull)
  scope: acr
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.acrPull)
    principalId: identityPrincipalId
    principalType: 'ServicePrincipal'
  }
}

// ---------------------------------------------------------------------------
// Storage (hierarchical namespace, RBAC only, no shared keys)
// ---------------------------------------------------------------------------
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  tags: tags
  kind: 'StorageV2'
  sku: {
    name: 'Standard_LRS'
  }
  properties: {
    isHnsEnabled: true
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    publicNetworkAccess: 'Enabled'
    accessTier: 'Hot'
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
  properties: {}
}

resource blobContainers 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = [for containerName in blobContainerNames: {
  parent: blobService
  name: containerName
  properties: {
    publicAccess: 'None'
  }
}]

resource storageIdentityRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, identityPrincipalId, roleIds.storageBlobDataContributor)
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.storageBlobDataContributor)
    principalId: identityPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource storageDeployerRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(deployerObjectId)) {
  name: guid(storage.id, deployerObjectId, roleIds.storageBlobDataContributor)
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.storageBlobDataContributor)
    principalId: deployerObjectId
  }
}

// ---------------------------------------------------------------------------
// Key Vault (RBAC, soft delete on; purge protection stays off, which means the
// property is omitted because the API only accepts the value true)
// ---------------------------------------------------------------------------
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  tags: tags
  properties: {
    tenantId: tenant().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    publicNetworkAccess: 'Enabled'
  }
}

resource keyVaultIdentityRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, identityPrincipalId, roleIds.keyVaultSecretsUser)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.keyVaultSecretsUser)
    principalId: identityPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource keyVaultDeployerRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(deployerObjectId)) {
  name: guid(keyVault.id, deployerObjectId, roleIds.keyVaultSecretsOfficer)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleIds.keyVaultSecretsOfficer)
    principalId: deployerObjectId
  }
}

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------
resource vnet 'Microsoft.Network/virtualNetworks@2024-05-01' = {
  name: vnetName
  location: location
  tags: tags
  properties: {
    addressSpace: {
      addressPrefixes: [
        '10.10.0.0/16'
      ]
    }
    subnets: [
      {
        name: 'snet-aca'
        properties: {
          addressPrefix: '10.10.0.0/23'
          delegations: [
            {
              name: 'Microsoft.App.environments'
              properties: {
                serviceName: 'Microsoft.App/environments'
              }
            }
          ]
        }
      }
      {
        name: 'snet-pg'
        properties: {
          addressPrefix: '10.10.2.0/24'
          delegations: [
            {
              name: 'Microsoft.DBforPostgreSQL.flexibleServers'
              properties: {
                serviceName: 'Microsoft.DBforPostgreSQL/flexibleServers'
              }
            }
          ]
        }
      }
    ]
  }
}

resource snetAca 'Microsoft.Network/virtualNetworks/subnets@2024-05-01' existing = {
  parent: vnet
  name: 'snet-aca'
}

resource snetPg 'Microsoft.Network/virtualNetworks/subnets@2024-05-01' existing = {
  parent: vnet
  name: 'snet-pg'
}

resource pgDnsZone 'Microsoft.Network/privateDnsZones@2024-06-01' = {
  name: pgDnsZoneName
  location: 'global'
  tags: tags
}

resource pgDnsLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2024-06-01' = {
  parent: pgDnsZone
  name: '${vnetName}-link'
  location: 'global'
  tags: tags
  properties: {
    virtualNetwork: {
      id: vnet.id
    }
    registrationEnabled: false
  }
}

// ---------------------------------------------------------------------------
// PostgreSQL Flexible Server (private access, Entra-only authentication)
// ---------------------------------------------------------------------------
resource postgres 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = {
  name: pgServerName
  location: location
  tags: tags
  sku: {
    name: 'Standard_B1ms'
    tier: 'Burstable'
  }
  properties: {
    version: '16'
    storage: {
      storageSizeGB: 32
      autoGrow: 'Disabled'
    }
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: 'Disabled'
    }
    highAvailability: {
      mode: 'Disabled'
    }
    network: {
      delegatedSubnetResourceId: snetPg.id
      privateDnsZoneArmResourceId: pgDnsZone.id
      publicNetworkAccess: 'Disabled'
    }
    authConfig: {
      activeDirectoryAuth: 'Enabled'
      passwordAuth: 'Disabled'
      tenantId: tenant().tenantId
    }
  }
  dependsOn: [
    pgDnsLink
  ]
}

// Entra administrators are created one after the other: the server rejects
// concurrent operations.
resource postgresAdminIdentity 'Microsoft.DBforPostgreSQL/flexibleServers/administrators@2024-08-01' = if (deployPostgresAdministrators) {
  parent: postgres
  name: identityPrincipalId
  properties: {
    principalType: 'ServicePrincipal'
    principalName: identityName
    tenantId: tenant().tenantId
  }
}

resource postgresAdminUser 'Microsoft.DBforPostgreSQL/flexibleServers/administrators@2024-08-01' = if (deployPostgresAdministrators && !empty(postgresAdminObjectId)) {
  parent: postgres
  name: postgresAdminObjectId
  properties: {
    principalType: 'User'
    principalName: postgresAdminPrincipalName
    tenantId: tenant().tenantId
  }
  dependsOn: [
    postgresAdminIdentity
  ]
}

resource postgresDatabase 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2024-08-01' = {
  parent: postgres
  name: pgDatabaseName
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
  dependsOn: [
    postgresAdminIdentity
    postgresAdminUser
  ]
}

// ---------------------------------------------------------------------------
// Container Apps environment (workload profiles, Consumption only, VNet-integrated)
// ---------------------------------------------------------------------------
resource containerAppsEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: containerAppsEnvName
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
    ]
    vnetConfiguration: {
      infrastructureSubnetId: snetAca.id
      internal: false
    }
    zoneRedundant: false
  }
}

var extractInternalUrl = 'http://kairos-extract.internal.${containerAppsEnv.properties.defaultDomain}'
var predictInternalUrl = 'http://kairos-predict.internal.${containerAppsEnv.properties.defaultDomain}'

// Environment variables shared by every app and job (config/app.yaml).
var commonEnv = [
  { name: 'KAIROS_ENV', value: environmentName }
  { name: 'ALLOW_REAL_NOTES_TO_LLM', value: allowRealNotesToLlm ? 'true' : 'false' }
  { name: 'KAIROS_STORAGE_ACCOUNT_URL', value: storageAccountUrl }
  { name: 'KAIROS_PG_HOST', value: postgres.properties.fullyQualifiedDomainName }
  { name: 'KAIROS_PG_DB', value: pgDatabaseName }
  { name: 'KAIROS_PG_USER', value: identityName }
  { name: 'KAIROS_PG_AUTH', value: 'entra' }
  { name: 'KAIROS_OPENAI_ENDPOINT', value: openAiEndpoint }
  { name: 'KAIROS_OPENAI_API_VERSION', value: openAiApiVersion }
  { name: 'KAIROS_OPENAI_EXTRACT_DEPLOYMENT', value: openAiExtractDeploymentName }
  { name: 'KAIROS_OPENAI_PRESCREEN_DEPLOYMENT', value: openAiPrescreenDeploymentName }
  { name: 'KAIROS_OPENAI_ADJUDICATE_DEPLOYMENT', value: openAiAdjudicateDeploymentName }
  { name: 'KAIROS_OPENAI_EXTRACT_REASONING_EFFORT', value: openAiExtractReasoningEffort }
  { name: 'KAIROS_OPENAI_ADJUDICATE_REASONING_EFFORT', value: openAiAdjudicateReasoningEffort }
  { name: 'KAIROS_KEYVAULT_URL', value: keyVaultUrl }
  { name: 'KAIROS_EXTRACT_URL', value: extractInternalUrl }
  { name: 'KAIROS_PREDICT_URL', value: predictInternalUrl }
  { name: 'KAIROS_MODEL_BLOB_PREFIX', value: 'latest' }
  { name: 'KAIROS_GIT_SHA', value: gitSha }
  { name: 'AZURE_CLIENT_ID', value: identityClientId }
  { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsights.properties.ConnectionString }
]

var appIdentity = {
  type: 'UserAssigned'
  userAssignedIdentities: {
    '${identity.id}': {}
  }
}

var acrRegistries = [
  {
    server: acr.properties.loginServer
    identity: identity.id
  }
]

var appResources = {
  cpu: json('0.5')
  memory: '1Gi'
}

var appScale = {
  minReplicas: 0
  maxReplicas: 2
}

// HTTP probes are attached once a real image is supplied; the public placeholder
// image serves neither the port nor the health path.
func httpProbes(path string, port int) array => [
  {
    type: 'Liveness'
    httpGet: {
      path: path
      port: port
    }
    initialDelaySeconds: 15
    periodSeconds: 30
    timeoutSeconds: 5
    failureThreshold: 3
  }
  {
    type: 'Readiness'
    httpGet: {
      path: path
      port: port
    }
    initialDelaySeconds: 5
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 3
  }
]

// ---------------------------------------------------------------------------
// Container Apps
// ---------------------------------------------------------------------------
resource extractApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'kairos-extract'
  location: location
  tags: union(tags, { 'azd-service-name': 'extract' })
  identity: appIdentity
  properties: {
    environmentId: containerAppsEnv.id
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: false
        targetPort: 8000
        allowInsecure: true
        transport: 'auto'
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
      registries: acrRegistries
    }
    template: {
      containers: [
        {
          name: 'kairos-extract'
          image: extractImageEffective
          resources: appResources
          env: commonEnv
          probes: empty(extractImage) ? [] : httpProbes('/healthz', 8000)
        }
      ]
      scale: appScale
    }
  }
  dependsOn: [
    acrPullRole
  ]
}

resource predictApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'kairos-predict'
  location: location
  tags: union(tags, { 'azd-service-name': 'predict' })
  identity: appIdentity
  properties: {
    environmentId: containerAppsEnv.id
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: false
        targetPort: 8000
        allowInsecure: true
        transport: 'auto'
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
      registries: acrRegistries
    }
    template: {
      containers: [
        {
          name: 'kairos-predict'
          image: predictImageEffective
          resources: appResources
          env: commonEnv
          probes: empty(predictImage) ? [] : httpProbes('/healthz', 8000)
        }
      ]
      scale: appScale
    }
  }
  dependsOn: [
    acrPullRole
  ]
}

resource demoApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'kairos-demo'
  location: location
  tags: union(tags, { 'azd-service-name': 'demo' })
  identity: appIdentity
  properties: {
    environmentId: containerAppsEnv.id
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8501
        allowInsecure: false
        transport: 'auto'
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
      registries: acrRegistries
      secrets: demoAuthEnabled ? [
        {
          name: demoAuthSecretName
          keyVaultUrl: demoAuthSecretUrl
          identity: identity.id
        }
      ] : []
    }
    template: {
      containers: [
        {
          name: 'kairos-demo'
          image: demoImageEffective
          resources: appResources
          env: commonEnv
          probes: empty(demoImage) ? [] : httpProbes('/_stcore/health', 8501)
        }
      ]
      scale: appScale
    }
  }
  dependsOn: [
    acrPullRole
    keyVaultIdentityRole
  ]
}

// Built-in authentication (Entra ID) for the demo, enabled once the app
// registration exists and its client secret is in Key Vault.
resource demoAuth 'Microsoft.App/containerApps/authConfigs@2024-03-01' = if (demoAuthEnabled) {
  parent: demoApp
  name: 'current'
  properties: {
    platform: {
      enabled: true
    }
    globalValidation: {
      unauthenticatedClientAction: 'RedirectToLoginPage'
      redirectToProvider: 'azureactivedirectory'
    }
    identityProviders: {
      azureActiveDirectory: {
        enabled: true
        registration: {
          openIdIssuer: '${environment().authentication.loginEndpoint}${tenant().tenantId}/v2.0'
          clientId: demoAuthClientId
          clientSecretSettingName: demoAuthSecretName
        }
        validation: {
          allowedAudiences: [
            demoAuthClientId
            'api://${demoAuthClientId}'
          ]
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Container Apps Jobs (manual trigger). The jobs image ENTRYPOINT is
// `python services/jobs/cli.py`; args select the command.
// ---------------------------------------------------------------------------
var jobDefinitions = [
  {
    name: 'kairos-job-scenarios'
    args: [ 'scenarios', '--all' ]
    azdServiceName: 'jobs'
  }
  {
    name: 'kairos-job-train'
    args: [ 'train' ]
    azdServiceName: ''
  }
  {
    name: 'kairos-job-evaluate'
    args: [ 'evaluate', '--all' ]
    azdServiceName: ''
  }
]

resource jobs 'Microsoft.App/jobs@2024-03-01' = [for job in jobDefinitions: {
  name: job.name
  location: location
  tags: empty(job.azdServiceName) ? tags : union(tags, { 'azd-service-name': job.azdServiceName })
  identity: appIdentity
  properties: {
    environmentId: containerAppsEnv.id
    workloadProfileName: 'Consumption'
    configuration: {
      triggerType: 'Manual'
      replicaTimeout: 7200
      replicaRetryLimit: 0
      manualTriggerConfig: {
        parallelism: 1
        replicaCompletionCount: 1
      }
      registries: acrRegistries
    }
    template: {
      containers: [
        {
          name: job.name
          image: jobsImageEffective
          args: job.args
          resources: {
            cpu: json('2')
            memory: '4Gi'
          }
          env: commonEnv
        }
      ]
    }
  }
  dependsOn: [
    acrPullRole
  ]
}]

// ---------------------------------------------------------------------------
// Cost Management budget on the resource group
// ---------------------------------------------------------------------------
resource budget 'Microsoft.Consumption/budgets@2023-11-01' = if (!empty(ownerEmail)) {
  name: budgetName
  properties: {
    category: 'Cost'
    amount: budgetAmount
    timeGrain: 'Monthly'
    timePeriod: {
      startDate: budgetStartDate
    }
    notifications: {
      Actual_GreaterThan_80_Percent: {
        enabled: true
        operator: 'GreaterThan'
        threshold: 80
        thresholdType: 'Actual'
        contactEmails: [
          ownerEmail
        ]
      }
      Forecasted_GreaterThan_100_Percent: {
        enabled: true
        operator: 'GreaterThan'
        threshold: 100
        thresholdType: 'Forecasted'
        contactEmails: [
          ownerEmail
        ]
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------
output acrName string = acr.name
output acrLoginServer string = acr.properties.loginServer
output storageAccountName string = storage.name
output storageAccountUrl string = storageAccountUrl
output keyVaultName string = keyVault.name
output keyVaultUrl string = keyVaultUrl
output postgresFqdn string = postgres.properties.fullyQualifiedDomainName
output postgresServerName string = postgres.name
output postgresDatabaseName string = pgDatabaseName
output containerAppsEnvironmentName string = containerAppsEnv.name
output environmentDefaultDomain string = containerAppsEnv.properties.defaultDomain
output demoFqdn string = demoApp.properties.configuration.ingress.fqdn
output extractInternalUrl string = extractInternalUrl
output predictInternalUrl string = predictInternalUrl
output appInsightsConnectionString string = appInsights.properties.ConnectionString
output logAnalyticsWorkspaceName string = logAnalytics.name
output extractImage string = extractImageEffective
output predictImage string = predictImageEffective
output demoImage string = demoImageEffective
output jobsImage string = jobsImageEffective
output demoAuthClientId string = demoAuthClientId
