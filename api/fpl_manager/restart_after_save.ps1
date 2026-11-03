# Restarts each manager run just after its next policy save, so the new code
# is loaded and almost no play is lost. A run that has not saved within
# $MaxWaitMinutes (it may be hung) is restarted anyway; it resumes from its
# last save.
param([int]$MaxWaitMinutes = 50)

$api = Split-Path -Parent $PSScriptRoot
$runs = @(
    @{ Name = 'model_v2_log_minbands_blend25'; Set = '35'; Extra = '' },
    @{ Name = 'model_ORACLE_v2_log_real_minutes'; Set = '34'; Extra = '--allow-oracle' },
    @{ Name = 'model_v1_h5'; Set = '29'; Extra = '' }
)
$log = Join-Path $api 'fpl_manager_runs\restart_after_save.log'
function Say($text) { Add-Content -Path $log -Value ('{0} {1}' -f (Get-Date -Format 'HH:mm:ss'), $text) }

foreach ($run in $runs) {
    $run.Rng = Join-Path $api ('fpl_manager_runs\{0}\rng.pkl' -f $run.Name)
    $run.Saved = (Get-Item $run.Rng).LastWriteTime
    $run.Done = $false
}
Say ('Watching {0} runs for their next save' -f $runs.Count)
$deadline = (Get-Date).AddMinutes($MaxWaitMinutes)

while (@($runs | Where-Object { -not $_.Done }).Count -gt 0) {
    foreach ($run in $runs | Where-Object { -not $_.Done }) {
        $saved = (Get-Item $run.Rng).LastWriteTime
        $timedOut = (Get-Date) -gt $deadline
        if ($saved -le $run.Saved -and -not $timedOut) { continue }
        if ($timedOut) { Say ('{0}: no save within {1} min, restarting anyway' -f $run.Name, $MaxWaitMinutes) }
        else { Say ('{0}: saved at {1:HH:mm:ss}' -f $run.Name, $saved) }
        Start-Sleep 15
        $checkpoint = 'fpl_manager_runs/' + $run.Name
        $all = Get-CimInstance Win32_Process
        $roots = $all | Where-Object {
            $_.Name -eq 'cmd.exe' -and $_.CommandLine -match 'run_manager_model\.cmd' -and
            $_.CommandLine -match [regex]::Escape($checkpoint)
        }
        foreach ($root in $roots) {
            & taskkill.exe /T /F /PID $root.ProcessId | Out-Null
            Say ('{0}: stopped process tree {1}' -f $run.Name, $root.ProcessId)
        }
        Start-Sleep 5
        $command = 'cmd.exe /c ""{0}" {1} {2} {3}"' -f (Join-Path $PSScriptRoot 'run_manager_model.cmd'), $run.Set, $checkpoint, $run.Extra
        $made = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
            CommandLine = $command; CurrentDirectory = $PSScriptRoot
        }
        Say ('{0}: relaunched, PID {1}, return {2}' -f $run.Name, $made.ProcessId, $made.ReturnValue)
        $run.Done = $true
    }
    Start-Sleep 10
}
Say 'All runs restarted'
