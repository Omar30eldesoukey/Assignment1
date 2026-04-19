param(
    [int]$MaxPdfs = 5,
    [int]$TopK = 6,
    [switch]$SkipInstall,
    [switch]$NoApp
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

$venvDir = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue

if ($pythonCommand) {
    $bootstrapPyExe = $pythonCommand.Source
    $bootstrapPrefixArgs = @()
} elseif ($pyLauncher) {
    $bootstrapPyExe = $pyLauncher.Source
    $bootstrapPrefixArgs = @("-3")
} else {
    throw "Python not found. Install Python 3.11+ and retry."
}

$venvCreated = $false
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating local virtual environment at: $venvDir"
    & $bootstrapPyExe @bootstrapPrefixArgs -m venv $venvDir
    $venvCreated = $true
}

if (-not (Test-Path $venvPython)) {
    throw "Failed to create virtual environment at $venvDir"
}

$pyExe = $venvPython
$pyPrefixArgs = @()

Write-Host "Project root: $projectRoot"
Write-Host "Python: $pyExe $($pyPrefixArgs -join ' ')"

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args,
        [Parameter(Mandatory = $true)]
        [string]$StepName
    )

    & $pyExe @pyPrefixArgs @Args
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

if ($SkipInstall -and $venvCreated) {
    Write-Host "SkipInstall was requested, but a new virtual environment was created. Installing requirements once."
}

if ((-not $SkipInstall) -or $venvCreated) {
    Write-Host "[1/4] Installing requirements..."
    Invoke-PythonChecked -Args @("-m", "pip", "install", "-r", "requirements.txt") -StepName "Dependency installation"
} else {
    Write-Host "[1/4] Skipping requirements install (SkipInstall switch used)."
}

Write-Host "[2/4] Building index..."
Invoke-PythonChecked -Args @("scripts/build_index.py", "--pdf-dir", "data/raw", "--image-dir", "data/processed/images", "--index-dir", "data/index", "--max-pdfs", "$MaxPdfs") -StepName "Index build"

Write-Host "[3/4] Running evaluation..."
Invoke-PythonChecked -Args @("scripts/run_evaluation.py", "--index-dir", "data/index", "--benchmark", "evaluation/benchmark_queries.json", "--top-k", "$TopK") -StepName "Evaluation"

if ($NoApp) {
    Write-Host "[4/4] App launch skipped (NoApp switch used)."
    exit 0
}

Write-Host "[4/4] Launching Streamlit app..."
Write-Host "Open: http://localhost:8501"
Write-Host "Press Ctrl+C in this terminal to stop the app."
Invoke-PythonChecked -Args @("-m", "streamlit", "run", "app.py", "--server.port", "8501") -StepName "Streamlit app launch"
