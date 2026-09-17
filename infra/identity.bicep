// KAIROS: user-assigned managed identity.
// Created before everything else so that its principalId can be handed to the
// OpenAI module (role assignment) and to the PostgreSQL Entra administrator
// (whose resource name must be the object id, a deploy-time constant).
targetScope = 'resourceGroup'

@description('Name of the user-assigned managed identity.')
param name string

@description('Azure region.')
param location string = resourceGroup().location

@description('Tags applied to the identity.')
param tags object = {}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: name
  location: location
  tags: tags
}

output name string = identity.name
output id string = identity.id
output clientId string = identity.properties.clientId
output principalId string = identity.properties.principalId
