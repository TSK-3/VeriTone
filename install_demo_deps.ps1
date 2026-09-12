$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { Write-Error "venv python not found at $py"; exit 1 }
Write-Output "Installing API/dev dependencies..."
& $py -m pip install --quiet fastapi "uvicorn[standard]" pytest
if ($LASTEXITCODE -ne 0) { Write-Output "FAILED_API_DEPS"; exit 1 }
Write-Output "Installing CPU torch..."
& $py -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { Write-Output "FAILED_TORCH"; exit 1 }
Write-Output "INSTALL_DONE"
& $py -m pip list