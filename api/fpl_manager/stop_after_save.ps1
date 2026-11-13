# Stops each v3 manager run just after its next policy save, so almost no play
# is lost. Start them again with run_manager_v3.cmd (or run_three_v3.cmd): each
# continues from its checkpoint. A run that has not saved within
# $MaxWaitMinutes (it may be hung) is stopped anyway; it resumes from its last
# save.
# -Names picks other checkpoint folders under fpl_manager_runs (default: the three paused runs).
param(
    [int]$MaxWaitMinutes = 40,
    [string[]]$Names = @('v3_v2_log_minbands_blend25', 'v3_ORACLE_v2_log_real_minutes', 'v3_v1_h5')
)

$api = Split-Path -Parent $PSScriptRoot
$runs = @($Names | ForEach-Object { @{ Name = $_ } })
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
        # Either slash: run_three_v3.cmd passes fpl_manager_runs/..., others fpl_manager_runs\...
        $checkpoint = 'fpl_manager_runs[\\/]' + [regex]::Escape($run.Name)
        $all = Get-CimInstance Win32_Process
        $roots = @($all | Where-Object {
            $_.Name -eq 'cmd.exe' -and $_.CommandLine -match 'run_manager_v3\.cmd' -and
            $_.CommandLine -match ($checkpoint + '(\s|"|$)')
        })
        if (-not $roots) {
            # Started with "call run_manager_v3.cmd" from another .cmd (such as
            # swap_b0_for_lgbm_seed1.cmd): no cmd.exe names the checkpoint, so stop
            # the nearest cmd.exe above the run's python processes instead.
            $byId = @{}
            foreach ($p in $all) { $byId[[int]$p.ProcessId] = $p }
            $pythons = $all | Where-Object {
                $_.Name -eq 'python.exe' -and $_.CommandLine -match '-m fpl_manager' -and
                $_.CommandLine -match ('--checkpoint\s+"?' + $checkpoint + '(\s|"|$)')
            }
            $roots = @(foreach ($p in $pythons) {
                $up = $byId[[int]$p.ParentProcessId]
                while ($up -and $up.Name -ne 'cmd.exe') { $up = $byId[[int]$up.ParentProcessId] }
                if ($up) { $up }
            }) | Sort-Object ProcessId -Unique
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
