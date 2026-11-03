# Stops each v3 manager run just after its next policy save, so almost no play
# is lost. Start them again with run_manager_v3.cmd (or run_three_v3.cmd): each
# continues from its checkpoint. A run that has not saved within
# $MaxWaitMinutes (it may be hung) is stopped anyway; it resumes from its last
# save.
param([int]$MaxWaitMinutes = 40)

$api = Split-Path -Parent $PSScriptRoot
$runs = @(
    @{ Name = 'v3_v2_log_minbands_blend25' },
    @{ Name = 'v3_ORACLE_v2_log_real_minutes' },
    @{ Name = 'v3_v1_h5' }
)
$log = Join-Path $api 'fpl_manager_runs\stop_after_save.log'
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
        if ($timedOut) { Say ('{0}: no save within {1} min, stopping anyway' -f $run.Name, $MaxWaitMinutes) }
        else { Say ('{0}: saved at {1:HH:mm:ss}' -f $run.Name, $saved) }
        # Let the save and the range cache's batched writes finish.
        Start-Sleep 15
        $checkpoint = 'fpl_manager_runs/' + $run.Name
        $all = Get-CimInstance Win32_Process
        $roots = $all | Where-Object {
            $_.Name -eq 'cmd.exe' -and $_.CommandLine -match 'run_manager_v3\.cmd' -and
            $_.CommandLine -match ([regex]::Escape($checkpoint) + '(\s|"|$)')
        }
        foreach ($root in $roots) {
            & taskkill.exe /T /F /PID $root.ProcessId | Out-Null
            Say ('{0}: stopped process tree {1}' -f $run.Name, $root.ProcessId)
        }
        if (-not $roots) { Say ('{0}: no running process found' -f $run.Name) }
        $run.Done = $true
    }
    Start-Sleep 10
}
Say 'All runs stopped'
