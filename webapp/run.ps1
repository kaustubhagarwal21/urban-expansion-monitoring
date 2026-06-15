# One-command launcher for the demo.
# Frees the ports, starts backend + frontend each in its own window, then opens
# the browser once the frontend is ready. Run from anywhere:
#   cd "C:\Users\KAUSTUBH\Desktop\AISD PROJECT\webapp"; .\run.ps1
$root = $PSScriptRoot

# The project's Python deps (torch, fastapi, timm...) live in the Windows Store
# Python 3.11, NOT Anaconda base. Use it explicitly so a conda shell can't break it.
$py = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe"
if (-not (Test-Path $py)) { $py = "python" }  # fallback

# 1. Free ports 8000 + 5173 so stale processes never cause "address in use".
Write-Host "==> Clearing ports 8000 and 5173..." -ForegroundColor DarkGray
Get-NetTCPConnection -LocalPort 8000, 5173 -State Listen -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty OwningProcess -Unique |
  ForEach-Object { try { Stop-Process -Id $_ -Force -ErrorAction Stop } catch {} }
Start-Sleep -Milliseconds 400

# 2. Backend in its own window.
Write-Host "==> Starting backend  -> http://127.0.0.1:8000" -ForegroundColor Cyan
Start-Process powershell -WorkingDirectory "$root\backend" -ArgumentList @(
  "-NoExit", "-Command",
  "$py -m uvicorn main:app --host 127.0.0.1 --port 8000"
)

# 3. Frontend in its own window. Host/port come from vite.config.ts, so no flags.
Write-Host "==> Starting frontend -> http://127.0.0.1:5173" -ForegroundColor Cyan
Start-Process powershell -WorkingDirectory "$root\frontend" -ArgumentList @(
  "-NoExit", "-Command",
  "npm run dev"
)

# 4. Wait for the frontend to respond, then open the browser.
Write-Host "`n==> Waiting for the app (backend loads torch first, ~15-20s)..." -ForegroundColor DarkGray
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
  try { Invoke-WebRequest "http://127.0.0.1:5173/" -UseBasicParsing -TimeoutSec 1 | Out-Null; $ready = $true; break }
  catch { Start-Sleep -Seconds 1 }
}
if ($ready) {
  Write-Host "==> Frontend is up. Opening http://127.0.0.1:5173/" -ForegroundColor Green
  Start-Process "http://127.0.0.1:5173/"
} else {
  Write-Host "==> Frontend is slow to start - once its window shows 'Local: http://127.0.0.1:5173/', open that URL." -ForegroundColor Yellow
}

Write-Host "`nTwo server windows are now running. Keep them open while using the app." -ForegroundColor Green
Write-Host "To stop everything later: .\stop.ps1  (or just close the two windows)." -ForegroundColor Green
