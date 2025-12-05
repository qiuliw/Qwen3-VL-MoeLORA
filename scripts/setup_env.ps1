<#
.SYNOPSIS
  Unified environment bootstrapper for Windows PowerShell.

.PARAMETER TorchVersion
  PyTorch wheel version (default 2.5.1+cu124).

.PARAMETER TorchVisionVersion
  TorchVision wheel version (default 0.20.1+cu124).

.PARAMETER TorchIndexUrl
  Index URL for CUDA wheels (default https://download.pytorch.org/whl/cu124).

.PARAMETER TopK
  Number of samples to keep when converting CSV to JSON (default 500).

.PARAMETER SkipData
  Switch to skip dataset download/convert.
#>

param(
    [string]$TorchVersion = "2.5.1+cu124",
    [string]$TorchVisionVersion = "0.20.1+cu124",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu124",
    [int]$TopK = 500,
    [switch]$SkipData
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[setup_env] UV not detected, installing via pip..."
    python -m pip install --upgrade pip
    python -m pip install uv
}

Write-Host "[setup_env] Creating virtual environment..."
uv venv
& .\.venv\Scripts\Activate.ps1

Write-Host "[setup_env] Installing CUDA wheels torch $TorchVersion / torchvision $TorchVisionVersion"
uv pip install --index-url $TorchIndexUrl `
    "torch==$TorchVersion" `
    "torchvision==$TorchVisionVersion"

Write-Host "[setup_env] Installing project requirements..."
uv pip install -r requirements.txt
uv pip install -r multi_agent\requirements.txt

if (-not $SkipData) {
    Write-Host "[setup_env] Downloading COCO captions (TopK=$TopK)..."
    uv run python download_data2csv.py --output ./coco_2014_caption/train.csv
    uv run python csv2json.py `
        --csv ./coco_2014_caption/train.csv `
        --json ./coco_2014_caption/train.json `
        --top_k $TopK
} else {
    Write-Host "[setup_env] SkipData flag present -> skipping dataset download."
}

Write-Host ""
Write-Host "[setup_env] Done!"
Write-Host "Next steps:"
Write-Host "  1. uv run python test.py --model ./qwen3-vl-4b-instruct --image ./image/demo.jpeg --prompt `"描述这张图片`""
Write-Host "  2. uv run python MoeLORA.py --model ./qwen3-vl-4b-instruct --train_json ./coco_2014_caption/train.json --output_dir ./output/Qwen3-VL-4Blora"
Write-Host "Remember to run `uv run python -c `"import swanlab; swanlab.login(api_key='YOUR_KEY')`"` once if you plan to train with SwanLab."

