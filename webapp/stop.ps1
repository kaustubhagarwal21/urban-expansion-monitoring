# Stop the demo servers by freeing ports 8000 (backend) and 5173 (frontend).
# Targets only whatever is listening on those ports, so it won't touch
# unrelated python/node processes.

$ports = 8000, 5173
foreach ($port in $ports) {
  $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
  if (-not $conns) {
    Write-Host "Port $port : nothing running." -ForegroundColor DarkGray
    continue
  }
  $procIds = $conns.OwningProcess | Sort-Object -Unique
  foreach ($procId in $procIds) {
    try {
      $p = Get-Process -Id $procId -ErrorAction Stop
      Stop-Process -Id $procId -Force -ErrorAction Stop
      Write-Host ("Port {0} : stopped {1} (PID {2})." -f $port, $p.ProcessName, $procId) -ForegroundColor Green
    } catch {
      Write-Host ("Port {0} : could not stop PID {1} - {2}" -f $port, $procId, $_.Exception.Message) -ForegroundColor Yellow
    }
  }
}

Write-Host "`nDone. Ports 8000 and 5173 are now free - safe to run .\run.ps1" -ForegroundColor Cyan
