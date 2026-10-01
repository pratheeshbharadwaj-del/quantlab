#requires -Version 7.0
<#
.SYNOPSIS
  Deploys QuantLab to Azure Container Apps.

.DESCRIPTION
  1. Creates (or reuses) a resource group.
  2. Deploys infra/main.bicep (Container Apps Environment, ACR, managed
     identity, storage account + Azure Files share, and two Container Apps
     provisioned with a placeholder image).
  3. Builds the backend and frontend Docker images directly in the cloud via
     `az acr build` (no local Docker required).
  4. Patches both Container Apps to use the real images.
  5. Re-deploys once more with the backend's real public URL as the allowed
     CORS origin's counterpart (QNT_API_URL) so frontend <-> backend calls work.

.PARAMETER SubscriptionId
  Azure subscription ID to deploy into. Required.

.PARAMETER ResourceGroupName
  Resource group to create/use. Defaults to "quantlab-rg".

.PARAMETER Location
  Azure region for compute resources (Container Apps Environment, ACR).
  Defaults to "eastus2". If this region lacks quota/SKU availability for your
  subscription, check `az provider show --namespace Microsoft.App` /
  `az account list-locations` and pick another region, then re-run.

.EXAMPLE
  ./deploy.ps1 -SubscriptionId 00000000-0000-0000-0000-000000000000
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$SubscriptionId,

    [string]$ResourceGroupName = "quantlab-rg",

    [string]$Location = "eastus2",

    [string]$EnvironmentName = "prod"
)

$ErrorActionPreference = "Stop"

function Assert-LastExitCode($step) {
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Step failed: $step (exit code $LASTEXITCODE)"
        exit 1
    }
}

Write-Host "== Checking Azure CLI login ==" -ForegroundColor Cyan
az account show *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Not logged in. Launching 'az login'..." -ForegroundColor Yellow
    az login
    Assert-LastExitCode "az login"
}

Write-Host "== Setting active subscription ($SubscriptionId) ==" -ForegroundColor Cyan
az account set --subscription $SubscriptionId
Assert-LastExitCode "az account set"

Write-Host "== Creating resource group '$ResourceGroupName' in '$Location' ==" -ForegroundColor Cyan
az group create --name $ResourceGroupName --location $Location | Out-Null
Assert-LastExitCode "az group create"

$repoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "== Deploying infrastructure (infra/main.bicep) ==" -ForegroundColor Cyan
$deployOutput = az deployment group create `
    --resource-group $ResourceGroupName `
    --template-file "$repoRoot/infra/main.bicep" `
    --parameters environmentName=$EnvironmentName location=$Location storageLocation=$Location `
    --query "properties.outputs" -o json
Assert-LastExitCode "az deployment group create"
$outputs = $deployOutput | ConvertFrom-Json

$registryLoginServer = $outputs.registryLoginServer.value
$registryName = $registryLoginServer.Split('.')[0]
$backendAppName = (az deployment group show --resource-group $ResourceGroupName --name main --query "properties.outputResources" -o tsv 2>$null)
# Resource names aren't returned directly as outputs beyond URLs; re-derive app names by listing container apps in the RG.
$backendAppName = az containerapp list --resource-group $ResourceGroupName --query "[?contains(name, 'cab')].name" -o tsv
$frontendAppName = az containerapp list --resource-group $ResourceGroupName --query "[?contains(name, 'caf')].name" -o tsv

Write-Host "== Building backend image via ACR Tasks ==" -ForegroundColor Cyan
az acr build --registry $registryName --image quantlab-backend:latest "$repoRoot/backend"
Assert-LastExitCode "az acr build (backend)"

Write-Host "== Building frontend image via ACR Tasks ==" -ForegroundColor Cyan
az acr build --registry $registryName --image quantlab-frontend:latest "$repoRoot/frontend"
Assert-LastExitCode "az acr build (frontend)"

Write-Host "== Updating backend Container App to use real image ==" -ForegroundColor Cyan
az containerapp update --name $backendAppName --resource-group $ResourceGroupName --image "$registryLoginServer/quantlab-backend:latest"
Assert-LastExitCode "az containerapp update (backend)"

$backendUrl = $outputs.backendUrl.value
Write-Host "== Updating frontend Container App to use real image + backend URL ==" -ForegroundColor Cyan
az containerapp update --name $frontendAppName --resource-group $ResourceGroupName --image "$registryLoginServer/quantlab-frontend:latest" --set-env-vars "QNT_API_URL=$backendUrl"
Assert-LastExitCode "az containerapp update (frontend)"

$frontendUrl = $outputs.frontendUrl.value
Write-Host "== Restricting backend CORS to the frontend's public URL ==" -ForegroundColor Cyan
az containerapp update --name $backendAppName --resource-group $ResourceGroupName --set-env-vars "QNT_ALLOWED_ORIGINS=$frontendUrl"
Assert-LastExitCode "az containerapp update (backend CORS)"
az containerapp ingress cors enable --name $backendAppName --resource-group $ResourceGroupName --allowed-origins $frontendUrl --allowed-methods "*" --allowed-headers "*" | Out-Null

Write-Host ""
Write-Host "== Deployment complete ==" -ForegroundColor Green
Write-Host "Frontend: $frontendUrl"
Write-Host "Backend:  $backendUrl"
Write-Host ""
Write-Host "Next: open the frontend URL, import/upload a model, and confirm /health on the backend responds." -ForegroundColor Yellow
