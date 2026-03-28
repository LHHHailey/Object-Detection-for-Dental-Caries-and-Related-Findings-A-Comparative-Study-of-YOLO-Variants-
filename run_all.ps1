#Requires -Version 5.1
<#
.SYNOPSIS
  One-click pipeline: extract dataset -> pip deps -> convert -> optional single-epoch smoke train -> optional predict -> optional web.

.PARAMETER ProjectRoot
  Project folder containing run_convert.py, object_detector.py, etc.

.PARAMETER TarPath
  Path to dentalai *.tar (DatasetNinja layout: train/valid/test + meta.json).

.PARAMETER SkipExtract
  Skip tar extract; use existing .\dataset

.PARAMETER SkipTrain
  Skip post-convert smoke training.

.PARAMETER SkipPredict
  Skip predict on caries.jpg after train.

.PARAMETER SkipWeb
  Do not start object_detector.py in background.

.PARAMETER Device
  ultralytics device: cpu | 0 | 0,1,...
#>
param(
    [string]$ProjectRoot = $PSScriptRoot,
    [string]$TarPath = "D:\Desktop\YOLOv8\dentalai_sample.tar",
    [switch]$SkipExtract,
    [switch]$SkipTrain,
    [switch]$SkipPredict,
    [switch]$SkipWeb,
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logFile = Join-Path $logDir "run_all_$ts.log"

function Write-Log([string]$msg) {
    $line = "$(Get-Date -Format o)  $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line -Encoding UTF8
}

try {
    Start-Transcript -Path (Join-Path $logDir "run_all_${ts}_transcript.txt") -Force
} catch {
    Write-Log "WARN: Transcript start failed: $_"
}

Write-Log "=== run_all.ps1 start ==="
Write-Log "ProjectRoot=$ProjectRoot"
Write-Log "TarPath=$TarPath  SkipExtract=$SkipExtract  SkipTrain=$SkipTrain"

if (-not (Test-Path -LiteralPath $ProjectRoot)) {
    Write-Log "ERROR: ProjectRoot not found: $ProjectRoot"
    exit 1
}
Set-Location -LiteralPath $ProjectRoot

# --- Python ---
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Log "ERROR: python not on PATH"
    exit 1
}
Write-Log "Python: $($py.Source)"
& python --version 2>&1 | ForEach-Object { Write-Log $_ }

# --- Step 1: extract dataset ---
if (-not $SkipExtract) {
    if (-not (Test-Path -LiteralPath $TarPath)) {
        Write-Log "ERROR: Tar not found: $TarPath"
        exit 1
    }
    $tmp = Join-Path $ProjectRoot "_dataset_extract_tmp"
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    New-Item -ItemType Directory -Path $tmp | Out-Null
    Write-Log "Extracting tar -> $tmp"
    & tar -xf $TarPath -C $tmp
    if ($LASTEXITCODE -ne 0) {
        Write-Log "ERROR: tar extract failed, exit $LASTEXITCODE"
        exit 1
    }
    $needTrain = Test-Path (Join-Path $tmp "train")
    $needValid = Test-Path (Join-Path $tmp "valid")
    if (-not ($needTrain -and $needValid)) {
        Write-Log "ERROR: tar must contain train/ and valid/ at top level after extract"
        exit 1
    }
    $ds = Join-Path $ProjectRoot "dataset"
    if (Test-Path $ds) { Remove-Item -Recurse -Force $ds }
    New-Item -ItemType Directory -Path $ds | Out-Null
    foreach ($name in @("train", "valid", "test", "meta.json", "README.md", "LICENSE.md")) {
        $src = Join-Path $tmp $name
        if (Test-Path $src) {
            Move-Item -LiteralPath $src -Destination (Join-Path $ds $name)
        }
    }
    Remove-Item -Recurse -Force $tmp
    Write-Log "dataset ready under $ds"
} else {
    if (-not (Test-Path (Join-Path $ProjectRoot "dataset\meta.json"))) {
        Write-Log "ERROR: SkipExtract set but dataset\meta.json missing"
        exit 1
    }
    Write-Log "SkipExtract: using existing dataset"
}

# --- Step 2: pip install ---
Write-Log "pip install -r requirements.txt"
& python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Log "ERROR: pip install failed"
    exit 1
}

# --- Step 3: convert ---
$conv = Join-Path $ProjectRoot "run_convert.py"
if (-not (Test-Path $conv)) {
    Write-Log "ERROR: run_convert.py not found"
    exit 1
}
Write-Log "python run_convert.py"
& python $conv
if ($LASTEXITCODE -ne 0) {
    Write-Log "ERROR: run_convert.py failed"
    exit 1
}
$dataYaml = Join-Path $ProjectRoot "yolo_dataset\data.yaml"
if (-not (Test-Path $dataYaml)) {
    Write-Log "ERROR: yolo_dataset\data.yaml missing after convert"
    exit 1
}

# --- Step 4: smoke train (1 epoch, same repo style) ---
if (-not $SkipTrain) {
    Write-Log "Smoke train 1 epoch yolov8m (device=$Device)"
    $trainCmd = @"
from ultralytics import YOLO
import os
m = YOLO('yolov8m.pt')
m.train(data=os.path.join(os.getcwd(),'yolo_dataset','data.yaml'), model='yolov8m.pt', epochs=1, imgsz=640, batch=16, device='$Device', project='runs', name='run_all_smoke', exist_ok=True)
"@
    & python -c $trainCmd
    if ($LASTEXITCODE -ne 0) {
        Write-Log "ERROR: smoke train failed"
        exit 1
    }
}

# --- Step 5: predict ---
if (-not $SkipPredict) {
    $weights = Join-Path $ProjectRoot "runs\detect\runs\run_all_smoke\weights\best.pt"
    if (-not (Test-Path $weights)) {
        $weights = Join-Path $ProjectRoot "runs\detect\run_all_smoke\weights\best.pt"
    }
    if (-not (Test-Path $weights)) {
        Write-Log "WARN: smoke best.pt not found, skip predict"
    } else {
        Write-Log "Predict caries.jpg -> runs/detect/run_all_predict"
        $pred = @"
from ultralytics import YOLO
YOLO(r'$weights')('caries.jpg', save=True, project='runs', name='run_all_predict', exist_ok=True)
"@
        & python -c $pred
    }
}

# --- Step 6: web (background) ---
if (-not $SkipWeb) {
    Write-Log "Starting object_detector.py on :8080 (background job). Stop manually in Task Manager or close window."
    $job = Start-Job -ScriptBlock {
        Set-Location $using:ProjectRoot
        python object_detector.py
    }
    Write-Log "Background Job Id: $($job.Id)  (Get-Job / Stop-Job)"
    Start-Sleep -Seconds 4
    try {
        $code = (Invoke-WebRequest -Uri "http://127.0.0.1:8080" -UseBasicParsing -TimeoutSec 5).StatusCode
        Write-Log "HTTP / -> $code"
    } catch {
        Write-Log "WARN: could not reach http://127.0.0.1:8080 : $_"
    }
}

Write-Log "=== run_all.ps1 done. Log: $logFile ==="
try { Stop-Transcript } catch {}
