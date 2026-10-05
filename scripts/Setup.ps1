param([switch]$Experiments,[switch]$Neural,[switch]$Test)
$ErrorActionPreference='Stop'
$taskRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Push-Location -LiteralPath $taskRoot
try {
    $taskPython=Join-Path $taskRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $taskPython)) {
        if (Get-Command py -ErrorAction SilentlyContinue) { & py -3.12 -m venv .venv }
        else { & python -m venv .venv }
        if ($LASTEXITCODE -ne 0) { throw 'Please install CPython 3.12 first.' }
    }
    & $taskPython -c "import sys; assert sys.version_info[:2] == (3,12), 'Use Python 3.12 for the Windows dependency lock.'"
    if ($LASTEXITCODE -ne 0) { throw 'This Windows lock requires Python 3.12.' }
    & $taskPython -m pip install -r requirements/windows-base.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $taskPython -m pip install -e . --no-deps
    if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
    if ($Neural -or $Experiments) {
        & $taskPython -m pip install -r requirements/windows-neural.txt
        if ($LASTEXITCODE -ne 0) { throw 'Neural repair dependencies failed.' }
    }
    if ($Experiments) {
        & $taskPython -m pip install -r requirements/windows-experiments.txt
        if ($LASTEXITCODE -ne 0) { throw 'Experiment dependencies failed.' }
    }
    if ($Test) {
        & $taskPython -m pip install -r requirements/windows-test.txt
        if ($LASTEXITCODE -ne 0) { throw 'Test dependencies failed.' }
    }
    Write-Host 'Setup complete. Run .venv/Scripts/python.exe -m xiyuan_mvp.gui'
} finally { Pop-Location }
