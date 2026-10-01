// QuantLab Azure Container Apps deployment.
//
// Deploys: Log Analytics workspace, Container Apps Environment, Azure
// Container Registry, a user-assigned managed identity (with AcrPull),
// a Storage Account + Azure Files share (persistent model storage mounted
// into the backend container app), and two Container Apps (backend, frontend).
//
// NOTE: Container Apps are provisioned with the placeholder image
// mcr.microsoft.com/azuredocs/containerapps-helloworld:latest so the
// infrastructure can be created before any application image exists in ACR.
// Run deploy.ps1 (which builds+pushes the real images via `az acr build`,
// then patches the container apps with `az containerapp update --image`)
// to get the actual QuantLab backend/frontend running.
targetScope = 'resourceGroup'

@description('Short environment name used to build the resource token (e.g. "prod", "demo").')
param environmentName string = 'prod'

@description('Location for the Container Apps Environment, Container Registry, and Container Apps.')
param location string = resourceGroup().location

@description('Location for the Storage Account (can differ from "location" if that region lacks storage capacity/SKU).')
param storageLocation string = location

@description('Comma-separated list of origins allowed to call the backend API (set to the frontend public URL after first deploy).')
param allowedOrigins string = '*'

var resourceToken = toLower(uniqueString(subscription().id, resourceGroup().id, location, environmentName))
var placeholderImage = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'

// ---------------------------------------------------------------------------
// Log Analytics (required for the Container Apps Environment)
// ---------------------------------------------------------------------------
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'azlog${resourceToken}'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// ---------------------------------------------------------------------------
// Container Registry
// ---------------------------------------------------------------------------
resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: 'azcr${resourceToken}'
  location: location
  sku: { name: 'Basic' }
  properties: {
    adminUserEnabled: false
  }
}

// ---------------------------------------------------------------------------
// User-assigned managed identity + AcrPull role assignment (one per registry)
// ---------------------------------------------------------------------------
resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'azid${resourceToken}'
  location: location
}

var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'

resource acrPullAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, identity.id, acrPullRoleId)
  scope: registry
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleId)
  }
}

// ---------------------------------------------------------------------------
// Storage Account + Azure Files share (persistent model storage)
// ---------------------------------------------------------------------------
resource storage 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: 'azst${resourceToken}'
  location: storageLocation
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    allowSharedKeyAccess: true // Container Apps Environment storage links currently require an account key; no other secrets are stored.
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
  }
}

resource fileServices 'Microsoft.Storage/storageAccounts/fileServices@2023-01-01' = {
  parent: storage
  name: 'default'
}

resource modelsShare 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-01-01' = {
  parent: fileServices
  name: 'quantlab-models'
  properties: {
    shareQuota: 512 // GiB; increase if you expect many/large model files.
  }
}

// ---------------------------------------------------------------------------
// Container Apps Environment
// ---------------------------------------------------------------------------
resource containerAppEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'azcae${resourceToken}'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

resource modelsStorageLink 'Microsoft.App/managedEnvironments/storages@2024-03-01' = {
  parent: containerAppEnv
  name: 'quantlabmodels'
  properties: {
    azureFile: {
      accountName: storage.name
      accountKey: storage.listKeys().keys[0].value
      shareName: modelsShare.name
      accessMode: 'ReadWrite'
    }
  }
}

// ---------------------------------------------------------------------------
// Backend Container App (FastAPI) — mounts the Azure Files share for model storage.
// ---------------------------------------------------------------------------
resource backendApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'azcab${resourceToken}'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        corsPolicy: {
          allowedOrigins: split(allowedOrigins, ',')
          allowedMethods: ['*']
          allowedHeaders: ['*']
        }
      }
      registries: [
        {
          server: registry.properties.loginServer
          identity: identity.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'backend'
          image: placeholderImage
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            { name: 'QNT_MODELS_DIR', value: '/data/models' }
            { name: 'QNT_ALLOWED_ORIGINS', value: allowedOrigins }
          ]
          volumeMounts: [
            { volumeName: 'models', mountPath: '/data/models' }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 10
              periodSeconds: 30
            }
          ]
        }
      ]
      volumes: [
        {
          name: 'models'
          storageType: 'AzureFile'
          storageName: modelsStorageLink.name
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 1 // Keep at 1: model state is loaded in-process memory, not shared across replicas.
      }
    }
  }
  dependsOn: [
    acrPullAssignment
  ]
}

// ---------------------------------------------------------------------------
// Frontend Container App (static React build served by nginx)
// ---------------------------------------------------------------------------
resource frontendApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'azcaf${resourceToken}'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 80
        transport: 'auto'
        corsPolicy: {
          allowedOrigins: ['*']
          allowedMethods: ['*']
          allowedHeaders: ['*']
        }
      }
      registries: [
        {
          server: registry.properties.loginServer
          identity: identity.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'frontend'
          image: placeholderImage
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            { name: 'QNT_API_URL', value: 'https://${backendApp.properties.configuration.ingress.fqdn}' }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
      }
    }
  }
  dependsOn: [
    acrPullAssignment
    backendApp
  ]
}

output backendUrl string = 'https://${backendApp.properties.configuration.ingress.fqdn}'
output frontendUrl string = 'https://${frontendApp.properties.configuration.ingress.fqdn}'
output registryLoginServer string = registry.properties.loginServer
output managedIdentityId string = identity.id
output resourceGroupName string = resourceGroup().name
