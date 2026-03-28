#Requires -Version 5.1
<#
  Wrapper: ensure cwd is project root, then run train_yolo_compare.py with logging.
  Example:
    .\run_yolo_compare.ps1 -Device 0
    .\run_yolo_compare.ps1 -Device cpu -Epochs 2   # quick test
#>
param(
    [string]$ProjectRoot = $PSScriptRoot,
    [string]$DataYaml = "yolo_dataset\data.yaml",
    [int]$Epochs = 100,
    [int]$Batch = 16,
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $ProjectRoot

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$log = Join-Path $logDir "compare_$ts.log"

function Log([string]$m) { "$(Get-Date -Format o) $m" | Tee-Object -FilePath $log -Append }

Log "train_yolo_compare.py starting"
if (-not (Test-Path $DataYaml)) {
    Log "ERROR: $DataYaml not found. Run run_all.ps1 or run_convert.py first."
    exit 1
}

$pyargs = @(
    "train_yolo_compare.py",
    "--data", $DataYaml,
    "--epochs", "$Epochs",
    "--batch", "$Batch",
    "--device", $Device
)
Log "python $($pyargs -join ' ')"
& python @pyargs 2>&1 | ForEach-Object { Log $_ }
$code = $LASTEXITCODE
Log "exit code $code"
exit $code
