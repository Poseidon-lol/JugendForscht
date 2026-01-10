# Quick ORCA smoke test (water single-point B3LYP/def2-SVP)

$orcaExe = $env:ORCA_EXE
if (-not $orcaExe -and $env:ORCADIR) {
    $candidate = Join-Path $env:ORCADIR "orca.exe"
    if (Test-Path $candidate) {
        $orcaExe = $candidate
    }
}
if (-not $orcaExe) {
    $cmd = Get-Command orca.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        $orcaExe = $cmd.Source
    }
}
if (-not $orcaExe) {
    throw "ORCA executable not found. Set ORCA_EXE/ORCADIR or add orca.exe to PATH."
}
$work = Join-Path $env:TEMP "orca_smoke"
New-Item -ItemType Directory -Force $work | Out-Null

# Write input
@"
! SP B3LYP def2-SVP TightSCF
* xyz 0 1
O    0.0000  0.0000  0.0000
H    0.7586  0.0000  0.5043
H   -0.7586  0.0000  0.5043
*
"@ | Set-Content -Path (Join-Path $work "water.inp")

Push-Location $work
& $orcaExe ".\water.inp" | Tee-Object -FilePath "water.out"
$exit = $LASTEXITCODE
Pop-Location

Write-Host "ORCA exit code:" $exit
if (Test-Path (Join-Path $work "water.out")) {
    $energy = Select-String -Path (Join-Path $work "water.out") -Pattern "TOTAL SCF ENERGY"
    if ($energy) { Write-Host $energy }
    else { Write-Host "No TOTAL SCF ENERGY line found (check water.out for errors)." }
} else {
    Write-Host "water.out missing; ORCA may not have run."
}
