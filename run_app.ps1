# ============================================================
# ScholarIQ Platform Windows PowerShell Startup Script
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "       SCHOLARIQ — ACADEMIC INTELLIGENCE PLATFORM         " -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ScriptDir "AI_Student_performance_Analytics-main\Student-Performance-Prediction-System-main\.venv\Scripts\python.exe"
$VenvStreamlit = Join-Path $ScriptDir "AI_Student_performance_Analytics-main\Student-Performance-Prediction-System-main\.venv\Scripts\streamlit.exe"
$AppScript = Join-Path $ScriptDir "app.py"

if (-not (Test-Path $VenvStreamlit)) {
    Write-Host "Warning: Inner virtual environment not found. Attempting global streamlit..." -ForegroundColor Red
    $VenvStreamlit = "streamlit"
}

Write-Host "Running automated pre-flight test suite..." -ForegroundColor Green
& $VenvPython (Join-Path $ScriptDir "run_tests.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pre-flight tests failed! Halting application launch." -ForegroundColor Red
    exit 1
}

Write-Host "`nLaunching ScholarIQ Web Interface..." -ForegroundColor Cyan
& $VenvStreamlit run $AppScript --server.headless false
