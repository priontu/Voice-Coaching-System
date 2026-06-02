# run_demo.ps1 — VocalCoach demo pipeline (PowerShell)
#
# Runs the unified inference pipeline on a sample audio file,
# exports a JSON result, and generates a scoring dashboard plot.
#
# Usage:
#   .\run_demo.ps1
#   .\run_demo.ps1 -AudioFile "path\to\my_recording.wav"
#   .\run_demo.ps1 -AudioFile "recording.wav" -MusicXML "score.musicxml"

param(
    [string]$AudioFile   = "samples\example.wav",
    [string]$MusicXML    = "",
    [string]$TextGrid    = "",
    [string]$OutputDir   = "outputs\demo",
    [switch]$NoScores    = $false,
    [switch]$Verbose     = $false
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  VocalCoach Demo Pipeline" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# --- Validate environment first ---
Write-Host "Step 1/3  Checking environment..." -ForegroundColor Yellow
py "$ScriptRoot\scripts\validate_environment.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Environment check failed. Fix issues above, then re-run." -ForegroundColor Red
    exit 1
}

# --- Validate audio file ---
if (-not (Test-Path $AudioFile)) {
    Write-Host ""
    Write-Host "Audio file not found: $AudioFile" -ForegroundColor Red
    Write-Host "Create a samples\ directory and add a WAV file, or pass -AudioFile <path>" -ForegroundColor Yellow
    exit 1
}

# --- Build argument list ---
$Args = @(
    ".\inference\run_pipeline.py",
    "--audio", $AudioFile,
    "--output_dir", $OutputDir,
    "--export-json",
    "--plot"
)

if ($MusicXML -ne "") { $Args += @("--musicxml", $MusicXML) }
if ($TextGrid -ne "") { $Args += @("--textgrid", $TextGrid) }
if (-not $NoScores)   {
    $Args += "--compute-metrics"
    $Args += "--compute-scores"
}
if ($Verbose) { $Args += "--verbose" }

# --- Run pipeline ---
Write-Host ""
Write-Host "Step 2/3  Running inference pipeline..." -ForegroundColor Yellow
Write-Host "  Audio:     $AudioFile"
if ($MusicXML) { Write-Host "  MusicXML:  $MusicXML" }
if ($TextGrid)  { Write-Host "  TextGrid:  $TextGrid" }
Write-Host "  Output:    $OutputDir"
Write-Host ""

py @Args
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Pipeline failed (exit $LASTEXITCODE). Check logs above." -ForegroundColor Red
    exit $LASTEXITCODE
}

# --- Done ---
Write-Host ""
Write-Host "Step 3/3  Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Outputs saved to: $OutputDir" -ForegroundColor Green
Write-Host "  - JSON:   $OutputDir\*_unified.json"
Write-Host "  - Plots:  $OutputDir\*_dashboard.png"
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
