param(
    [string]$Config = "configs/train_conf_3d.yaml",
    [string]$Device = ""
)

$cmd = @("python", "src/main.py", "train-surrogate-3d", "--config", $Config)
if ($Device) {
    $cmd += @("--device", $Device)
}

& $cmd[0] @($cmd[1..($cmd.Length - 1)])
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
