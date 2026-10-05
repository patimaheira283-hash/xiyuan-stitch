param(
    [ValidateSet('status', 'push', 'logs', 'download')]
    [string]$Action = 'status',
    [Parameter(Mandatory=$true)][string]$Credentials,
    [string]$OutputDirectory = '',
    [string]$FilePattern = '',
    [switch]$Follow,
[ValidateSet('v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12','v15')][string]$Experiment = 'v1',
    [int]$TimeoutSeconds = 3600
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe'
$kaggleExe = Join-Path $projectRoot '.venv/Scripts/kaggle.exe'
$settings = Get-Content -LiteralPath $Credentials -Raw | ConvertFrom-Json
$oldUsername, $oldKey, $oldToken = $env:KAGGLE_USERNAME, $env:KAGGLE_KEY, $env:KAGGLE_API_TOKEN
try {
    $env:KAGGLE_USERNAME = $settings.username
    $env:KAGGLE_KEY = $settings.key
    $env:KAGGLE_API_TOKEN = $null
    $notebookRef = "$($settings.username)/xiyuan-stitch-research"
    Push-Location -LiteralPath $projectRoot
    try {
        switch ($Action) {
            'status' { & $kaggleExe kernels status $notebookRef }
            'logs' { if ($Follow) { & $kaggleExe kernels logs $notebookRef --follow } else { & $kaggleExe kernels logs $notebookRef } }
            'download' {
                if (-not $OutputDirectory) {
                    $OutputDirectory = Join-Path $projectRoot ('outputs/research/' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
                }
                if ($FilePattern) {
                    & $pythonExe -m scripts.download_research --kernel $notebookRef --output $OutputDirectory --pattern $FilePattern
                } else {
                    & $pythonExe -m scripts.download_research --kernel $notebookRef --output $OutputDirectory
                }
            }
            'push' {
                $current = & $kaggleExe kernels status $notebookRef 2>&1 | Out-String
                if ($current -match 'RUNNING|QUEUED') { throw "Research notebook is still running. Do not submit a duplicate." }
                if (-not (Test-Path 'data/research-cases.json')) { throw 'Missing pinned data/research-cases.json manifest.' }
                & $pythonExe -m scripts.build_research_notebook --owner $settings.username --experiment $Experiment
                if ($LASTEXITCODE -ne 0) { throw 'Notebook build failed.' }
                $metadata = Get-Content 'kaggle/research/kernel-metadata.json' -Raw | ConvertFrom-Json
                if ($metadata.is_private -ne $true -or $metadata.id -ne $notebookRef) { throw 'Notebook must be private and owned by the selected account.' }
                & $kaggleExe kernels push -p kaggle/research --accelerator NvidiaTeslaT4 --timeout $TimeoutSeconds
            }
        }
        if ($LASTEXITCODE -ne 0) { throw "Kaggle failed with exit $LASTEXITCODE." }
    } finally { Pop-Location }
} finally {
    $env:KAGGLE_USERNAME, $env:KAGGLE_KEY, $env:KAGGLE_API_TOKEN = $oldUsername, $oldKey, $oldToken
    $settings = $null
}
