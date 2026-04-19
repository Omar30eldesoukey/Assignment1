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

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue

if ($pythonCommand) {
    $pyExe = $pythonCommand.Source
    $pyPrefixArgs = @()
} elseif ($pyLauncher) {
    $pyExe = $pyLauncher.Source
    $pyPrefixArgs = @("-3")
} else {
    throw "Python not found. Install Python 3.11+ and retry."
}

Write-Host "Project root: $projectRoot"
Write-Host "Python: $pyExe $($pyPrefixArgs -join ' ')"

if (-not $SkipInstall) {
    Write-Host "[1/4] Installing requirements..."
    & $pyExe @pyPrefixArgs -m pip install -r requirements.txt
}

Write-Host "[2/4] Building index..."
& $pyExe @pyPrefixArgs scripts/build_index.py --pdf-dir data/raw --image-dir data/processed/images --index-dir data/index --max-pdfs $MaxPdfs

Write-Host "[3/4] Running evaluation..."
& $pyExe @pyPrefixArgs scripts/run_evaluation.py --index-dir data/index --benchmark evaluation/benchmark_queries.json --top-k $TopK

if ($NoApp) {
    Write-Host "[4/4] App launch skipped (NoApp switch used)."
    exit 0
}

Write-Host "[4/4] Launching Streamlit app..."
Write-Host "Open: http://localhost:8501"
Write-Host "Press Ctrl+C in this terminal to stop the app."
& $pyExe @pyPrefixArgs -m streamlit run app.py --server.port 8501
