#Requires -Version 5.1
param(
    [string]$ProjectRoot = $PSScriptRoot,
    [string]$Weights = "best.pt"
)
Set-Location -LiteralPath $ProjectRoot
$env:YOLO_MODEL_PATH = $Weights
Write-Host "YOLO_MODEL_PATH=$Weights"
Write-Host "Open http://127.0.0.1:8080"
python object_detector.py
