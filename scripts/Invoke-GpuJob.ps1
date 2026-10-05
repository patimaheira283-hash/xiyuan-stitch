param(
    [ValidateSet('push','status','download')][string]$Action = 'status',
    [Parameter(Mandatory=$true)][string]$Credentials,
    [string]$Job,
    [string]$OutputDirectory = 'outputs/gpu-job-return'
)
$ErrorActionPreference='Stop'
$taskRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskPython=Join-Path $taskRoot '.venv/Scripts/python.exe'
$taskKaggle=Join-Path $taskRoot '.venv/Scripts/kaggle.exe'
$settings=Get-Content -LiteralPath $Credentials -Raw | ConvertFrom-Json
$oldUser,$oldKey,$oldToken=$env:KAGGLE_USERNAME,$env:KAGGLE_KEY,$env:KAGGLE_API_TOKEN
try {
    $env:KAGGLE_USERNAME=$settings.username
    $env:KAGGLE_KEY=$settings.key
    $env:KAGGLE_API_TOKEN=$null
    $taskRef="$($settings.username)/xiyuan-user-gpu-job"
    Push-Location -LiteralPath $taskRoot
    try {
        switch ($Action) {
            'status' { & $taskKaggle kernels status $taskRef }
            'download' { & $taskKaggle kernels output $taskRef -p $OutputDirectory --file-pattern 'job-result\.zip$' }
            'push' {
                if (-not $Job) { throw 'Supply -Job with the ZIP exported from the desktop.' }
                $taskStatus=& $taskKaggle kernels status $taskRef 2>&1 | Out-String
                if ($taskStatus -match 'RUNNING|QUEUED') { throw 'A desktop GPU job is still running.' }
                $taskSlug='xiyuan-job-'+(Get-Date -Format 'yyyyMMdd-HHmmss')
                $uploadRoot=Join-Path $taskRoot ('.work/'+$taskSlug)
                New-Item -ItemType Directory -Path $uploadRoot | Out-Null
                Copy-Item -LiteralPath $Job -Destination (Join-Path $uploadRoot 'desktop-job.bin')
                @{id="$($settings.username)/$taskSlug";title=$taskSlug;licenses=@(@{name='other'})} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $uploadRoot 'dataset-metadata.json') -Encoding utf8
                & $taskKaggle datasets create -p $uploadRoot -q
                if ($LASTEXITCODE -ne 0) { throw 'Private job dataset upload failed.' }
                & $taskPython -m scripts.build_gpu_job_notebook $Job --owner $settings.username --dataset "$($settings.username)/$taskSlug"
                if ($LASTEXITCODE -ne 0) { throw 'Notebook build failed.' }
                & $taskKaggle kernels push -p kaggle/user-job --accelerator NvidiaTeslaT4 --timeout 1800
            }
        }
        if ($LASTEXITCODE -ne 0) { throw "Kaggle exited with $LASTEXITCODE" }
    } finally { Pop-Location }
} finally {
    $env:KAGGLE_USERNAME,$env:KAGGLE_KEY,$env:KAGGLE_API_TOKEN=$oldUser,$oldKey,$oldToken
    $settings=$null
}
