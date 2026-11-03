@echo off
rem Starts the three manager runs side by side, 2 worker processes each, with no terminal attached.
rem Starts are 2 minutes apart so the season loading does not peak at the same time.
rem Each run logs to its checkpoint's console.log. They run until stopped.
rem The oracle is fed the real minutes players went on to play, so it cheats: compare it as a ceiling only.
cd /d "%~dp0"
start "" /b cmd /c run_manager_model.cmd 35 fpl_manager_runs/model_v2_log_minbands_blend25
rem ping waits about 2 minutes; timeout refuses to run without a console.
ping -n 121 127.0.0.1 > nul
start "" /b cmd /c run_manager_model.cmd 34 fpl_manager_runs/model_ORACLE_v2_log_real_minutes --allow-oracle
ping -n 121 127.0.0.1 > nul
start "" /b cmd /c run_manager_model.cmd 29 fpl_manager_runs/model_v1_h5
