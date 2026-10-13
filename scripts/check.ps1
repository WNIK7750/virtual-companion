[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repositoryRoot
try {
    & uv lock --check
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & uv run --locked ruff check apps/backend/src tests scripts
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & uv run --locked --extra provider python -m pytest tests -q -o faulthandler_timeout=60
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & npm --prefix apps/desktop test
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & npm --prefix apps/desktop run lint
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & npm --prefix apps/desktop run build
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & git diff --check
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & git diff --cached --check
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
Write-Host 'Public source checks passed.'
