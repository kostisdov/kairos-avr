<#
.SYNOPSIS
    KAIROS: provision the Azure infrastructure, register the demo Entra app, build the
    container images in Azure Container Registry and wire them into the Container Apps.

.DESCRIPTION
    Windows PowerShell 5.1 compatible (no &&, no ternaries). Same sequence as
    scripts/deploy.sh; every step is idempotent and may be re-run:

      1. az account set; register the resource providers and wait for "Registered"
      2. az deployment sub create with infra/main.bicep + infra/main.bicepparam
         (outputs -> infra/outputs.local.json, git-ignored)
      3. Entra app registration kairos-demo-<env> for the demo login, assignment
         required, owner (+ optional users) assigned, client secret stored ONCE in
         Key Vault (never printed, never written to a file)
      4. az acr build of extract, predict, jobs and demo from an allow-listed staging
         directory that is privacy-scanned before anything is uploaded
      5. second deployment passing the image references and the demo client id
      6. print the demo URL and the internal URLs

    Run it from a PowerShell console: .\scripts\deploy.ps1

.PARAMETER EnvironmentName
    Environment name (dev). Resource group defaults to rg-kairos-<EnvironmentName>.
.PARAMETER Tag
    Image tag to build (default: short git sha, or a UTC timestamp yyyyMMddHHmm
    outside a git repository). With -SkipBuild: tag of existing images to wire in.
.PARAMETER AllowedUserObjectIds
    Extra Entra user object ids allowed to sign in to the demo (owner is always added).
.PARAMETER SkipInfra
    Skip the provisioning pass; needs infra/outputs.local.json from an earlier run.
    The final pass that wires images and authentication still runs.
.PARAMETER SkipBuild
    Skip the image build; re-uses -Tag or the images currently deployed.
.PARAMETER SkipAuth
    Skip the Entra app registration; re-uses the client id of the deployed auth
    configuration, if any.
.PARAMETER NoAuth
    Leave the demo publicly accessible with no login: skips the Entra app
    registration and removes any existing auth config from the demo container app.
    Overrides -SkipAuth.
#>
[CmdletBinding()]
param(
    [string]$EnvironmentName = 'dev',
    [string]$SubscriptionId = $env:AZURE_SUBSCRIPTION_ID,   # default: the signed-in account
    [string]$Location = 'swedencentral',
    [string]$ResourceGroup = '',
    [string]$Tag = '',
    [string]$OwnerObjectId = $env:KAIROS_OWNER_OBJECT_ID,   # default: the signed-in user
    [string[]]$AllowedUserObjectIds = @(),
    [switch]$SkipInfra,
    [switch]$SkipBuild,
    [switch]$SkipAuth,
    [switch]$NoAuth
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
$RepoRoot = Split-Path -Parent $PSScriptRoot
$TemplateFile = 'infra/main.bicep'
$ParamFile = 'infra/main.bicepparam'
$OutputsFile = Join-Path $RepoRoot 'infra\outputs.local.json'
$PlaceholderImage = 'mcr.microsoft.com/k8se/quickstart:latest'
$DemoSecretName = 'demo-auth-client-secret'
$DefaultAppRoleId = '00000000-0000-0000-0000-000000000000'
$Graph = 'https://graph.microsoft.com/v1.0'
$Arm = 'https://management.azure.com'
$Services = @('extract', 'predict', 'jobs', 'demo')
$Providers = @(
    'Microsoft.App',
    'Microsoft.ContainerRegistry',
    'Microsoft.DBforPostgreSQL',
    'Microsoft.KeyVault',
    'Microsoft.OperationalInsights',
    'Microsoft.Storage',
    'Microsoft.CognitiveServices',
    'Microsoft.ManagedIdentity'
)
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

if (-not $ResourceGroup) {
    $ResourceGroup = "rg-kairos-$EnvironmentName"
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Step([string]$Message) {
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Info([string]$Message) {
    Write-Host "    $Message"
}

# Runs az, captures stdout, throws on a non-zero exit code unless -AllowFailure
# (which returns $null instead).
function Invoke-Az {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [switch]$AllowFailure
    )
    $output = & az @Arguments
    $exit = $LASTEXITCODE
    if ($exit -ne 0) {
        if ($AllowFailure) {
            return $null
        }
        $shown = $Arguments[0..([Math]::Min(2, $Arguments.Count - 1))] -join ' '
        throw "az $shown failed with exit code $exit"
    }
    if ($null -eq $output) {
        return ''
    }
    return (@($output | ForEach-Object { "$_" }) -join "`n").Trim()
}

# Runs az with live console output (long-running commands).
function Invoke-AzStream {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & az @Arguments
    $exit = $LASTEXITCODE
    if ($exit -ne 0) {
        $shown = $Arguments[0..([Math]::Min(2, $Arguments.Count - 1))] -join ' '
        throw "az $shown failed with exit code $exit"
    }
}

function Remove-DemoAuthConfig {
    $exists = Invoke-Az -Arguments @('resource', 'show', '--resource-group', $ResourceGroup, '--resource-type', 'Microsoft.App/containerApps', '--name', 'kairos-demo', '--query', 'name', '-o', 'tsv') -AllowFailure
    if (-not $exists) {
        return
    }
    $url = "$Arm/subscriptions/$SubscriptionId/resourceGroups/$ResourceGroup/providers/Microsoft.App/containerApps/kairos-demo/authConfigs/current?api-version=2024-03-01"
    Invoke-Az -Arguments @('rest', '--method', 'delete', '--url', $url) -AllowFailure | Out-Null
}

function Test-PostgresServerExists([string]$ResourceGroupName) {
    $name = Invoke-Az -Arguments @(
        'postgres', 'flexible-server', 'list', '--resource-group', $ResourceGroupName,
        '--query', '[0].name', '-o', 'tsv'
    ) -AllowFailure
    return [bool]$name
}

function Get-DeploymentOutput([string]$Name) {
    if (-not (Test-Path -LiteralPath $OutputsFile)) {
        throw "$OutputsFile not found; run once without -SkipInfra"
    }
    $outputs = [System.IO.File]::ReadAllText($OutputsFile) | ConvertFrom-Json
    $entry = $outputs.PSObject.Properties[$Name]
    if ($null -eq $entry -or $null -eq $entry.Value) {
        return ''
    }
    $valueProperty = $entry.Value.PSObject.Properties['value']
    if ($null -eq $valueProperty -or $null -eq $valueProperty.Value) {
        return ''
    }
    return [string]$valueProperty.Value
}

function Add-Param([System.Collections.ArrayList]$Target, [string]$Name, [string]$Value) {
    if ($Value) {
        [void]$Target.Add("$Name=$Value")
    }
}

function Invoke-Deployment([string]$Label, [string[]]$ExtraParameters) {
    $name = "kairos-$EnvironmentName-$Label-" + (Get-Date).ToUniversalTime().ToString('yyyyMMddHHmmss')
    Write-Step "Deployment $name ($TemplateFile + $ParamFile)"
    if ($ExtraParameters -and $ExtraParameters.Count -gt 0) {
        Write-Info ("overrides: " + ($ExtraParameters -join ' '))
    }
    $azArgs = @(
        'deployment', 'sub', 'create',
        '--name', $name,
        '--location', $Location,
        '--template-file', $TemplateFile,
        '--parameters', $ParamFile,
        '--parameters', "environmentName=$EnvironmentName", "resourceGroupName=$ResourceGroup", "location=$Location"
    )
    if ($ExtraParameters) {
        $azArgs += $ExtraParameters
    }
    $azArgs += @('--query', 'properties.outputs', '--output', 'json')
    $json = Invoke-Az -Arguments $azArgs
    if (-not $json) {
        throw "deployment $name returned no outputs"
    }
    [System.IO.File]::WriteAllText($OutputsFile, $json, $Utf8NoBom)
    Write-Info "outputs written to $OutputsFile"
}

function Register-Providers {
    Write-Step 'Registering resource providers'
    foreach ($ns in $Providers) {
        $state = Invoke-Az -Arguments @('provider', 'show', '--namespace', $ns, '--query', 'registrationState', '-o', 'tsv') -AllowFailure
        if ($state -ne 'Registered') {
            Invoke-Az -Arguments @('provider', 'register', '--namespace', $ns, '--output', 'none') | Out-Null
        }
    }
    foreach ($ns in $Providers) {
        $state = ''
        for ($attempt = 1; $attempt -le 60; $attempt++) {
            $state = Invoke-Az -Arguments @('provider', 'show', '--namespace', $ns, '--query', 'registrationState', '-o', 'tsv')
            if ($state -eq 'Registered') {
                break
            }
            Write-Info "${ns}: $state (waiting, attempt $attempt)"
            Start-Sleep -Seconds 10
        }
        if ($state -ne 'Registered') {
            throw "provider $ns did not reach Registered"
        }
        Write-Info "${ns}: Registered"
    }
}

# deployerObjectId override: the signed-in user; service principals keep the
# bicepparam default (they hold ARM roles only, see infra/README.md).
function Get-DeployerOverride {
    $type = Invoke-Az -Arguments @('account', 'show', '--query', 'user.type', '-o', 'tsv') -AllowFailure
    if ($type -eq 'user') {
        $oid = Invoke-Az -Arguments @('ad', 'signed-in-user', 'show', '--query', 'id', '-o', 'tsv') -AllowFailure
        if ($oid) {
            return "deployerObjectId=$oid"
        }
    }
    return $null
}

function Get-CurrentImage([string]$ResourceType, [string]$Name, [string[]]$Existing) {
    if ($Existing -notcontains $Name) {
        return ''
    }
    $image = Invoke-Az -Arguments @(
        'resource', 'show', '--resource-group', $ResourceGroup, '--resource-type', $ResourceType, '--name', $Name,
        '--api-version', '2024-03-01', '--query', 'properties.template.containers[0].image', '-o', 'tsv'
    ) -AllowFailure
    if (-not $image -or $image -eq $PlaceholderImage) {
        return ''
    }
    return $image
}

function Get-CurrentState {
    $state = @{ ExtractImage = ''; PredictImage = ''; DemoImage = ''; JobsImage = ''; AuthClientId = '' }
    $exists = Invoke-Az -Arguments @('group', 'exists', '--name', $ResourceGroup, '-o', 'tsv') -AllowFailure
    if ($exists -ne 'true') {
        return $state
    }
    $appsRaw = Invoke-Az -Arguments @('resource', 'list', '--resource-group', $ResourceGroup, '--resource-type', 'Microsoft.App/containerApps', '--query', '[].name', '-o', 'tsv') -AllowFailure
    $jobsRaw = Invoke-Az -Arguments @('resource', 'list', '--resource-group', $ResourceGroup, '--resource-type', 'Microsoft.App/jobs', '--query', '[].name', '-o', 'tsv') -AllowFailure
    $apps = @()
    $jobs = @()
    if ($appsRaw) { $apps = @($appsRaw -split "`n" | ForEach-Object { $_.Trim() }) }
    if ($jobsRaw) { $jobs = @($jobsRaw -split "`n" | ForEach-Object { $_.Trim() }) }
    $state.ExtractImage = Get-CurrentImage 'Microsoft.App/containerApps' 'kairos-extract' $apps
    $state.PredictImage = Get-CurrentImage 'Microsoft.App/containerApps' 'kairos-predict' $apps
    $state.DemoImage = Get-CurrentImage 'Microsoft.App/containerApps' 'kairos-demo' $apps
    $state.JobsImage = Get-CurrentImage 'Microsoft.App/jobs' 'kairos-job-scenarios' $jobs
    if ($apps -contains 'kairos-demo') {
        $url = "$Arm/subscriptions/$SubscriptionId/resourceGroups/$ResourceGroup/providers/Microsoft.App/containerApps/kairos-demo/authConfigs?api-version=2024-03-01"
        $clientId = Invoke-Az -Arguments @(
            'rest', '--method', 'get', '--url', $url,
            '--query', "value[?name=='current'].properties.identityProviders.azureActiveDirectory.registration.clientId", '-o', 'tsv'
        ) -AllowFailure
        if ($clientId) {
            $state.AuthClientId = (@($clientId -split "`n")[0]).Trim()
        }
    }
    return $state
}

# ---------------------------------------------------------------------------
# Step 3: Entra app registration for the demo login
# ---------------------------------------------------------------------------
function Invoke-Graph {
    param(
        [string]$Method,
        [string]$Url,
        [string]$Body = '',
        [string]$Query = '',
        [switch]$AllowFailure
    )
    $azArgs = @('rest', '--method', $Method, '--url', $Url)
    $tmp = $null
    if ($Body) {
        $tmp = [System.IO.Path]::GetTempFileName()
        [System.IO.File]::WriteAllText($tmp, $Body, $Utf8NoBom)
        $azArgs += @('--headers', 'Content-Type=application/json', '--body', "@$tmp")
    }
    if ($Query) {
        $azArgs += @('--query', $Query)
    }
    $azArgs += @('-o', 'tsv')
    try {
        return (Invoke-Az -Arguments $azArgs -AllowFailure:$AllowFailure)
    }
    finally {
        if ($tmp -and (Test-Path -LiteralPath $tmp)) {
            Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
        }
    }
}

function Test-PrincipalAssigned([string]$ServicePrincipalId, [string]$PrincipalId) {
    $assigned = Invoke-Graph -Method GET -Url "$Graph/servicePrincipals/$ServicePrincipalId/appRoleAssignedTo" -Query "value[?principalId=='$PrincipalId'].id"
    return [bool]$assigned
}

# Returns 'exists', 'missing' or 'error' (for example RBAC not propagated yet).
function Get-KeyVaultSecretState([string]$VaultName, [string]$SecretName) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & az keyvault secret show --vault-name $VaultName --name $SecretName --query id -o tsv 2>&1
        $exit = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previous
    }
    if ($exit -eq 0) {
        return 'exists'
    }
    $text = (@($out | ForEach-Object { "$_" }) -join "`n")
    if ($text -match 'SecretNotFound|was not found') {
        return 'missing'
    }
    Write-Warning $text
    return 'error'
}

function Ensure-DemoClientSecret([string]$KeyVaultName, [string]$AppId) {
    $state = 'error'
    for ($attempt = 1; $attempt -le 8; $attempt++) {
        $state = Get-KeyVaultSecretState $KeyVaultName $DemoSecretName
        if ($state -eq 'exists') {
            Write-Info "Key Vault secret $DemoSecretName exists; not rotating"
            return
        }
        if ($state -eq 'missing') {
            break
        }
        Write-Info "Key Vault not readable yet (role assignment propagating?); retry $attempt in 20 s"
        Start-Sleep -Seconds 20
    }
    if ($state -ne 'missing') {
        throw "cannot read Key Vault $KeyVaultName"
    }
    Write-Info "creating a client secret for $AppId and storing it in $KeyVaultName/$DemoSecretName"
    $secretValue = Invoke-Az -Arguments @('ad', 'app', 'credential', 'reset', '--id', $AppId, '--append', '--display-name', 'kairos-demo', '--years', '1', '--query', 'password', '-o', 'tsv')
    if (-not $secretValue) {
        throw 'credential reset returned no password'
    }
    try {
        Invoke-Az -Arguments @('keyvault', 'secret', 'set', '--vault-name', $KeyVaultName, '--name', $DemoSecretName, '--value', $secretValue, '--query', 'id', '-o', 'tsv') | Out-Null
    }
    finally {
        Remove-Variable -Name secretValue -ErrorAction SilentlyContinue
    }
    Write-Info 'stored'
}

function Ensure-DemoAppRegistration([string]$DemoFqdn, [string]$KeyVaultName) {
    $appName = "kairos-demo-$EnvironmentName"
    $redirectUri = "https://$DemoFqdn/.auth/login/aad/callback"
    Write-Step "Entra app registration $appName"
    if (-not $DemoFqdn) {
        throw 'demo FQDN unknown; cannot set the redirect URI'
    }

    $appId = Invoke-Az -Arguments @('ad', 'app', 'list', '--display-name', $appName, '--query', '[0].appId', '-o', 'tsv')
    if (-not $appId) {
        Write-Info "creating $appName"
        $appId = Invoke-Az -Arguments @(
            'ad', 'app', 'create', '--display-name', $appName, '--sign-in-audience', 'AzureADMyOrg',
            '--web-redirect-uris', $redirectUri, '--enable-id-token-issuance', 'true', '--query', 'appId', '-o', 'tsv'
        )
    }
    else {
        Write-Info "found $appName ($appId); refreshing the redirect URI"
        Invoke-Az -Arguments @('ad', 'app', 'update', '--id', $appId, '--web-redirect-uris', $redirectUri, '--enable-id-token-issuance', 'true', '--output', 'none') | Out-Null
    }
    if (-not $appId) {
        throw "could not determine the application id of $appName"
    }

    $spId = ''
    for ($attempt = 1; $attempt -le 6; $attempt++) {
        $spId = Invoke-Az -Arguments @('ad', 'sp', 'list', '--filter', "appId eq '$appId'", '--query', '[0].id', '-o', 'tsv')
        if ($spId) {
            break
        }
        $spId = Invoke-Az -Arguments @('ad', 'sp', 'create', '--id', $appId, '--query', 'id', '-o', 'tsv') -AllowFailure
        if ($spId) {
            break
        }
        Write-Info "service principal not ready yet (attempt $attempt); waiting 10 s"
        Start-Sleep -Seconds 10
    }
    if (-not $spId) {
        throw "could not create the service principal for $appId"
    }

    Write-Info "requiring assignment on the service principal $spId"
    Invoke-Graph -Method PATCH -Url "$Graph/servicePrincipals/$spId" -Body '{"appRoleAssignmentRequired": true}' | Out-Null

    $principals = @($OwnerObjectId) + @($AllowedUserObjectIds)
    foreach ($principal in $principals) {
        $principal = "$principal".Trim()
        if (-not $principal) {
            continue
        }
        if (Test-PrincipalAssigned $spId $principal) {
            Write-Info "assignment exists: $principal"
            continue
        }
        $body = '{"principalId":"' + $principal + '","resourceId":"' + $spId + '","appRoleId":"' + $DefaultAppRoleId + '"}'
        $result = Invoke-Graph -Method POST -Url "$Graph/servicePrincipals/$spId/appRoleAssignedTo" -Body $body -AllowFailure
        if ($null -ne $result) {
            Write-Info "assigned $principal"
        }
        elseif (Test-PrincipalAssigned $spId $principal) {
            # Graph answers 400 "Permission being assigned already exists" for duplicates.
            Write-Info "assignment exists: $principal"
        }
        else {
            throw "could not assign $principal to $appName"
        }
    }

    Ensure-DemoClientSecret $KeyVaultName $appId
    return $appId
}

# ---------------------------------------------------------------------------
# Step 4: build context and images
# ---------------------------------------------------------------------------
function Copy-IntoStaging([string]$Item, [string]$Staging) {
    $parent = Split-Path -Parent $Item
    $destinationDir = $Staging
    if ($parent) {
        $destinationDir = Join-Path $Staging $parent
    }
    if (-not (Test-Path -LiteralPath $destinationDir)) {
        New-Item -ItemType Directory -Path $destinationDir -Force | Out-Null
    }
    Copy-Item -LiteralPath $Item -Destination $destinationDir -Recurse -Force
}

function New-StagingDirectory {
    $staging = Join-Path ([System.IO.Path]::GetTempPath()) ('kairos-build-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $staging | Out-Null
    Write-Step "Staging the build context in $staging (allow-list only)"
    $required = @('pyproject.toml', 'src', 'services', 'config')
    $optional = @('README.md', 'requirements.lock.txt', '.dockerignore', 'data\reference', 'data\derived\aggregates', 'tests\fixtures\synthetic_notes')
    foreach ($item in $required) {
        if (-not (Test-Path -LiteralPath $item)) {
            throw "required build input missing: $item"
        }
        Copy-IntoStaging $item $staging
    }
    foreach ($item in $optional) {
        if (Test-Path -LiteralPath $item) {
            Copy-IntoStaging $item $staging
        }
        else {
            Write-Info "optional input absent, skipped: $item"
        }
    }
    # Never ship caches or build metadata.
    $junk = @(Get-ChildItem -LiteralPath $staging -Recurse -Force -Directory | Where-Object {
            $_.Name -eq '__pycache__' -or $_.Name -like '*.egg-info' -or $_.Name -eq '.pytest_cache' -or $_.Name -eq '.ruff_cache'
        })
    foreach ($dir in $junk) {
        if (Test-Path -LiteralPath $dir.FullName) {
            Remove-Item -LiteralPath $dir.FullName -Recurse -Force
        }
    }
    Get-ChildItem -LiteralPath $staging -Recurse -Force -File -Filter '*.pyc' | Remove-Item -Force
    # The allow-list cannot contain these; check anyway before anything leaves the machine.
    $forbidden = @(Get-ChildItem -LiteralPath $staging -Recurse -Force | Where-Object {
            $_.Extension -ieq '.xlsx' -or
            $_.FullName -like (Join-Path $staging 'data\raw*') -or
            $_.FullName -like (Join-Path $staging 'data\derived\private*') -or
            $_.FullName -like (Join-Path $staging 'tmp*')
        })
    if ($forbidden.Count -gt 0) {
        throw 'forbidden material found in the staging directory; nothing was uploaded'
    }
    return $staging
}

function Invoke-PrivacyScan([string]$Staging) {
    Write-Step 'Privacy scan of the staging directory'
    if (-not (Test-Path -LiteralPath 'scripts\privacy_scan.py')) {
        throw 'scripts/privacy_scan.py is missing'
    }
    & $Python 'scripts/privacy_scan.py' '--root' $Staging
    if ($LASTEXITCODE -ne 0) {
        throw 'privacy scan failed; nothing was uploaded'
    }
}

function Resolve-ImageTag {
    if ($Tag) {
        return $Tag
    }
    $sha = ''
    if ((Test-Path -LiteralPath '.git') -and (Get-Command git -ErrorAction SilentlyContinue)) {
        $previous = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $sha = (& git rev-parse --short HEAD 2>$null)
            if ($LASTEXITCODE -ne 0) {
                $sha = ''
            }
        }
        finally {
            $ErrorActionPreference = $previous
        }
    }
    if ($sha) {
        return ("$sha".Trim())
    }
    return (Get-Date).ToUniversalTime().ToString('yyyyMMddHHmm')
}

function Build-Images([string]$AcrName, [string]$Staging, [string]$ImageTag) {
    foreach ($svc in $Services) {
        if (-not (Test-Path -LiteralPath (Join-Path $Staging "services\$svc\Dockerfile"))) {
            throw "services/$svc/Dockerfile is missing"
        }
    }
    foreach ($svc in $Services) {
        Write-Step "az acr build kairos-${svc}:$ImageTag"
        Invoke-AzStream -Arguments @(
            'acr', 'build', '--registry', $AcrName, '--image', "kairos-${svc}:$ImageTag",
            '--file', "services/$svc/Dockerfile", '--build-arg', "GIT_SHA=$ImageTag", $Staging
        )
    }
}

# ---------------------------------------------------------------------------
# Main sequence
# ---------------------------------------------------------------------------
$stagingDir = ''
Push-Location $RepoRoot
try {
    if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
        throw 'Azure CLI (az) is required'
    }
    $Python = $null
    foreach ($candidate in @('python', 'python3')) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) {
            $Python = $command.Source
            break
        }
    }
    if (-not $Python) {
        throw 'python is required (privacy scan)'
    }

    if (-not $SubscriptionId) { $SubscriptionId = (az account show --query id -o tsv) }
    if (-not $OwnerObjectId) { $OwnerObjectId = (az ad signed-in-user show --query id -o tsv) }
    Write-Step "KAIROS deploy: env=$EnvironmentName resource-group=$ResourceGroup location=$Location subscription=$SubscriptionId"

    # 1. Subscription and providers
    Invoke-Az -Arguments @('account', 'set', '--subscription', $SubscriptionId) | Out-Null
    Register-Providers

    # Current state, so that a re-run never resets images or auth to placeholders.
    $current = Get-CurrentState
    $extractImage = $current.ExtractImage
    $predictImage = $current.PredictImage
    $demoImage = $current.DemoImage
    $jobsImage = $current.JobsImage
    $authClientId = $current.AuthClientId

    $commonParams = New-Object System.Collections.ArrayList
    $deployerOverride = Get-DeployerOverride
    if ($deployerOverride) {
        [void]$commonParams.Add($deployerOverride)
    }

    # 2. Provisioning pass
    if ($SkipInfra) {
        Write-Step 'Skipping the provisioning pass (-SkipInfra)'
        if (-not (Test-Path -LiteralPath $OutputsFile)) {
            throw "$OutputsFile not found; run once without -SkipInfra"
        }
    }
    else {
        $provisionParams = New-Object System.Collections.ArrayList
        foreach ($p in $commonParams) { [void]$provisionParams.Add($p) }
        Add-Param $provisionParams 'extractImage' $extractImage
        Add-Param $provisionParams 'predictImage' $predictImage
        Add-Param $provisionParams 'demoImage' $demoImage
        Add-Param $provisionParams 'jobsImage' $jobsImage
        Add-Param $provisionParams 'demoAuthClientId' $authClientId
        if ($extractImage) {
            Add-Param $provisionParams 'gitSha' ($extractImage.Substring($extractImage.LastIndexOf(':') + 1))
        }
        # The PostgreSQL Entra administrators are created once, the first time the
        # server itself is created; Postgres runs CREATE ROLE on every deployment of
        # those resources and 42710s if the role already exists, so a re-run against
        # an already-provisioned server (e.g. after an earlier pass failed downstream)
        # must not redeclare them.
        if (Test-PostgresServerExists $ResourceGroup) {
            Write-Info 'PostgreSQL server already exists; not redeclaring its Entra administrators'
            Add-Param $provisionParams 'deployPostgresAdministrators' 'false'
        }
        Invoke-Deployment 'provision' ([string[]]$provisionParams.ToArray())
    }

    $acrName = Get-DeploymentOutput 'acrName'
    $acrLoginServer = Get-DeploymentOutput 'acrLoginServer'
    $keyVaultName = Get-DeploymentOutput 'keyVaultName'
    $demoFqdn = Get-DeploymentOutput 'demoFqdn'
    if (-not $acrName -or -not $acrLoginServer -or -not $keyVaultName) {
        throw "deployment outputs incomplete in $OutputsFile"
    }

    # 3. Demo authentication
    $appId = ''
    if ($NoAuth) {
        Write-Step 'Removing built-in authentication; the demo will be publicly accessible (-NoAuth)'
        Remove-DemoAuthConfig
    }
    elseif ($SkipAuth) {
        Write-Step 'Skipping the Entra app registration (-SkipAuth)'
        $appId = $authClientId
        if ($appId) {
            Write-Info "re-using the client id of the deployed auth configuration: $appId"
        }
        else {
            Write-Info 'no auth configuration deployed yet; the demo stays without built-in authentication'
        }
    }
    else {
        $appId = Ensure-DemoAppRegistration $demoFqdn $keyVaultName
    }

    # 4. Images
    $imageTag = ''
    if ($SkipBuild) {
        Write-Step 'Skipping the image build (-SkipBuild)'
        if ($Tag) {
            $imageTag = $Tag
            $extractImage = "$acrLoginServer/kairos-extract:$imageTag"
            $predictImage = "$acrLoginServer/kairos-predict:$imageTag"
            $jobsImage = "$acrLoginServer/kairos-jobs:$imageTag"
            $demoImage = "$acrLoginServer/kairos-demo:$imageTag"
            Write-Info "re-using tag $imageTag"
        }
        else {
            Write-Info 'keeping the images currently deployed'
        }
    }
    else {
        $imageTag = Resolve-ImageTag
        $stagingDir = New-StagingDirectory
        Invoke-PrivacyScan $stagingDir
        Build-Images $acrName $stagingDir $imageTag
        $extractImage = "$acrLoginServer/kairos-extract:$imageTag"
        $predictImage = "$acrLoginServer/kairos-predict:$imageTag"
        $jobsImage = "$acrLoginServer/kairos-jobs:$imageTag"
        $demoImage = "$acrLoginServer/kairos-demo:$imageTag"
    }

    # 5. Wire images and authentication in
    $finalParams = New-Object System.Collections.ArrayList
    foreach ($p in $commonParams) { [void]$finalParams.Add($p) }
    Add-Param $finalParams 'extractImage' $extractImage
    Add-Param $finalParams 'predictImage' $predictImage
    Add-Param $finalParams 'demoImage' $demoImage
    Add-Param $finalParams 'jobsImage' $jobsImage
    Add-Param $finalParams 'demoAuthClientId' $appId
    # The provisioning pass (or an earlier run whose outputs -SkipInfra reuses) already
    # created the PostgreSQL Entra administrators; Postgres rejects re-creating an
    # existing role, so this pass must not redeclare them.
    Add-Param $finalParams 'deployPostgresAdministrators' 'false'
    if ($imageTag) {
        Add-Param $finalParams 'gitSha' $imageTag
    }
    elseif ($extractImage) {
        Add-Param $finalParams 'gitSha' ($extractImage.Substring($extractImage.LastIndexOf(':') + 1))
    }
    if ($extractImage -or $predictImage -or $demoImage -or $jobsImage -or $appId) {
        Invoke-Deployment 'wire' ([string[]]$finalParams.ToArray())
    }
    else {
        Write-Step 'No images or authentication to wire in; the apps keep the placeholder image'
    }

    # 6. Summary
    $demoFqdn = Get-DeploymentOutput 'demoFqdn'
    Write-Step 'Done'
    Write-Info "demo URL:            https://$demoFqdn"
    Write-Info ("extract (internal):  " + (Get-DeploymentOutput 'extractInternalUrl'))
    Write-Info ("predict (internal):  " + (Get-DeploymentOutput 'predictInternalUrl'))
    Write-Info "registry:            $acrLoginServer"
    Write-Info "key vault:           $keyVaultName"
    Write-Info ("postgres:            " + (Get-DeploymentOutput 'postgresFqdn'))
    if ($appId) {
        Write-Info "demo auth client id: $appId"
    }
    else {
        Write-Info 'demo auth:           not configured (run without -SkipAuth)'
    }
    Write-Info "outputs:             $OutputsFile"
    exit 0
}
catch {
    Write-Host ''
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.ScriptStackTrace) {
        Write-Host $_.ScriptStackTrace
    }
    exit 1
}
finally {
    if ($stagingDir -and (Test-Path -LiteralPath $stagingDir)) {
        Remove-Item -LiteralPath $stagingDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    Pop-Location
}
