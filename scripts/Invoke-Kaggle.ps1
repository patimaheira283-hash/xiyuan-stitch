param(
    [ValidateSet('status', 'quota', 'push', 'logs', 'download')]
    [string]$Action = 'status',
    [string]$Credentials = (Join-Path $env:USERPROFILE '.kaggle/kaggle.json'),
    [string]$OutputDirectory = ''
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$kaggleExe = Join-Path $projectRoot '.venv\Scripts\kaggle.exe'
if (-not (Test-Path -LiteralPath $kaggleExe)) {
    throw 'Install the Kaggle extra in .venv: python -m pip install -e ".[kaggle]"'
}
$credentialPath = (Resolve-Path -LiteralPath $Credentials).Path
if ([IO.Path]::GetFileName($credentialPath) -ne 'kaggle.json') {
    throw 'The Kaggle CLI expects a file named kaggle.json.'
}
$settings = Get-Content -Raw -LiteralPath $credentialPath | ConvertFrom-Json
if (-not $settings.username -or -not $settings.key) {
    throw 'kaggle.json must contain username and key.'
}
$notebookRef = "$($settings.username)/xiyuan-stitch-gpu-mvp"
$previousConfigDir = $env:KAGGLE_CONFIG_DIR
$previousToken = $env:KAGGLE_API_TOKEN
$previousUsername = $env:KAGGLE_USERNAME
$previousKey = $env:KAGGLE_KEY
try {
    # Scope authentication to this process and to the explicitly supplied credential.
    $env:KAGGLE_CONFIG_DIR = Split-Path -Parent $credentialPath
    $env:KAGGLE_API_TOKEN = $null
    $env:KAGGLE_USERNAME = $settings.username
    $env:KAGGLE_KEY = $settings.key
    Push-Location -LiteralPath $projectRoot
    try {
        switch ($Action) {
            'quota' { & $kaggleExe quota }
            'status' { & $kaggleExe kernels status $notebookRef }
            'logs' { & $kaggleExe kernels logs $notebookRef }
            'download' {
                if (-not $OutputDirectory) {
                    $OutputDirectory = Join-Path $projectRoot ('outputs\kaggle\' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
                }
                & $kaggleExe kernels output $notebookRef -p $OutputDirectory
            }
            'push' {
                & $pythonExe -m scripts.build_kaggle_notebook --owner $settings.username
                if ($LASTEXITCODE -ne 0) { throw 'Notebook build failed.' }
                $metadata = Get-Content -Raw -LiteralPath (Join-Path $projectRoot 'kaggle\kernel-metadata.json') | ConvertFrom-Json
                if ($metadata.is_private -ne $true -or $metadata.id -ne $notebookRef) {
                    throw 'Refusing to submit a public notebook or a different account notebook.'
                }
                & $kaggleExe kernels push -p kaggle --accelerator NvidiaTeslaT4 --timeout 1800
            }
        }
        if ($LASTEXITCODE -ne 0) { throw "Kaggle command failed (exit $LASTEXITCODE)." }
    }
    finally { Pop-Location }
}
finally {
    $env:KAGGLE_CONFIG_DIR = $previousConfigDir
    $env:KAGGLE_API_TOKEN = $previousToken
    $env:KAGGLE_USERNAME = $previousUsername
    $env:KAGGLE_KEY = $previousKey
    $settings = $null
}
