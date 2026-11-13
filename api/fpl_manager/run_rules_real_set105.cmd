@echo off
rem 2026-11-13, Hal asked: rules manager (default thresholds) on the real
rem 2023/24-2025/26, set 105, known calendar - as the training run's real-season
rem checks - to compare with the live Fargate manager (live_baseline_set105).
rem Log: fpl_manager_runs\rules_real_set105\console.log
cd /d "%~dp0.."
set DATABASE_PATH=/data/model_eval.db
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OUT=fpl_manager_runs\rules_real_set105
if not exist "%OUT%" mkdir "%OUT%"
echo ===== rules on real seasons, set 105, started %DATE% %TIME% ===== >> "%OUT%\console.log"
"..\venv\Scripts\python.exe" -m synthetic_seasons.manager_check --managers rules --prediction-set-id 105 --seasons 23,24,25 --copies "" --out "%OUT%\rules.csv" >> "%OUT%\console.log" 2>&1
echo ===== exit %ERRORLEVEL% %DATE% %TIME% ===== >> "%OUT%\console.log"
