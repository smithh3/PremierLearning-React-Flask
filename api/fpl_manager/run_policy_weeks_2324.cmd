@echo off
rem 2026-11-13, Hal asked: does the policy bank transfers on a season it trains on
rem and scores well? Plays the current v3_lgbm_v6_copies_fixed policy (top choice)
rem over the real 2023/24, set 105, known calendar, and writes every week.
rem Log: fpl_manager_runs\policy_weeks_2324\console.log
cd /d "%~dp0.."
set DATABASE_PATH=/data/model_eval.db
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OUT=fpl_manager_runs\policy_weeks_2324
if not exist "%OUT%" mkdir "%OUT%"
echo ===== policy week by week on 2023/24, started %DATE% %TIME% ===== >> "%OUT%\console.log"
"..\venv\Scripts\python.exe" -m fpl_manager.play_live_season --set-id 105 --season 23 --last 38 --calendar known --managers policy --checkpoint fpl_manager_runs\v3_lgbm_v6_copies_fixed --out "%OUT%" >> "%OUT%\console.log" 2>&1
echo ===== exit %ERRORLEVEL% %DATE% %TIME% ===== >> "%OUT%\console.log"
