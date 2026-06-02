@echo off
REM run_demo.bat — VocalCoach demo pipeline (Windows CMD)
REM
REM Usage:
REM   run_demo.bat
REM   run_demo.bat "path\to\recording.wav"
REM
REM For full options, use run_demo.ps1 or inference\run_pipeline.py directly.

setlocal

set AUDIO=%~1
if "%AUDIO%"=="" set AUDIO=samples\example.wav

set OUTPUT_DIR=outputs\demo

echo.
echo ============================================================
echo   VocalCoach Demo Pipeline
echo ============================================================
echo.

echo [1/3] Checking environment...
py scripts\validate_environment.py
if errorlevel 1 (
    echo.
    echo Environment check failed. Fix issues above, then re-run.
    exit /b 1
)

if not exist "%AUDIO%" (
    echo.
    echo Audio file not found: %AUDIO%
    echo Add a WAV file to samples\ or pass the path as the first argument.
    exit /b 1
)

echo.
echo [2/3] Running inference pipeline...
echo   Audio:   %AUDIO%
echo   Output:  %OUTPUT_DIR%
echo.

py inference\run_pipeline.py ^
    --audio "%AUDIO%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --compute-metrics ^
    --compute-scores ^
    --export-json ^
    --plot

if errorlevel 1 (
    echo.
    echo Pipeline failed. Check logs above.
    exit /b 1
)

echo.
echo [3/3] Complete!
echo   Outputs saved to: %OUTPUT_DIR%
echo.
echo ============================================================
