# AMFTA full-study watchdog.
# If results/run_main.log has not been updated in >5 minutes, the study is
# stalled/dead -> relaunch run_full_study.py (resumable; skips done configs).
$ErrorActionPreference = "SilentlyContinue"
$repo = "C:\Users\adeel\.gemini\antigravity\scratch\amfta-fl\amfta-fl"
$log  = Join-Path $repo "results\run_main.log"
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

$stale = $true
if (Test-Path $log) {
    $age = (New-TimeSpan -Start (Get-Item $log).LastWriteTime -End (Get-Date)).TotalMinutes
    if ($age -lt 5) { $stale = $false }
}

if ($stale) {
    Add-Content (Join-Path $repo "watchdog.log") "$stamp  STALE -> relaunching study"
    $py = "python"
    Start-Process -FilePath $py `
        -ArgumentList "experiments\run_focused_study.py" `
        -WorkingDirectory $repo `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $repo "full_study_watchdog.out") `
        -RedirectStandardError  (Join-Path $repo "full_study_watchdog.err")
} else {
    Add-Content (Join-Path $repo "watchdog.log") "$stamp  OK (log fresh)"
}
