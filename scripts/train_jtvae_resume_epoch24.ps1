param(
    [string]$Config = "configs/gen_conf_resume.yaml",
    [string]$Device = ""
)

$cmd = @("python", "src/main.py", "train-generator", "--config", $Config)
if ($Device) {
    $cmd += @("--device", $Device)
}

& $cmd[0] @($cmd[1..($cmd.Length - 1)])
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
