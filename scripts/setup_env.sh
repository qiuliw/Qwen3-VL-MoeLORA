#!/usr/bin/env bash

# Unified environment bootstrapper for Linux/macOS.
# Installs UV, creates a venv, pins PyTorch CUDA wheels, installs project deps,
# and optionally downloads data/model assets.

set -euo pipefail

TORCH_VERSION=${TORCH_VERSION:-"2.5.1+cu124"}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-"0.20.1+cu124"}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-"https://download.pytorch.org/whl/cu124"}
PYTHON_BIN=${PYTHON_BIN:-"python3"}
UV_BIN=${UV_BIN:-"uv"}
TOP_K=${TOP_K:-"500"}
SKIP_DATA=${SKIP_DATA:-"0"}

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "[setup_env] UV not found, installing via pip..."
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install uv
fi

echo "[setup_env] Creating virtual environment with ${UV_BIN}..."
"${UV_BIN}" venv
source .venv/bin/activate

echo "[setup_env] Installing CUDA wheels: torch ${TORCH_VERSION}, torchvision ${TORCHVISION_VERSION}"
"${UV_BIN}" pip install --index-url "${TORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}"

echo "[setup_env] Installing project requirements..."
"${UV_BIN}" pip install -r requirements.txt
"${UV_BIN}" pip install -r multi_agent/requirements.txt

if [[ "${SKIP_DATA}" != "1" ]]; then
  echo "[setup_env] Downloading COCO captions (TOP_K=${TOP_K})..."
  "${UV_BIN}" run python download_data2csv.py --output ./coco_2014_caption/train.csv
  "${UV_BIN}" run python csv2json.py \
    --csv ./coco_2014_caption/train.csv \
    --json ./coco_2014_caption/train.json \
    --top_k "${TOP_K}"
else
  echo "[setup_env] SKIP_DATA=1 -> skipping dataset download/convert steps."
fi

cat <<'EOF'
[setup_env] Done!
Next steps:
  1. uv run python test.py --model ./qwen3-vl-4b-instruct --image ./image/demo.jpeg --prompt "描述这张图片"
  2. uv run python MoeLORA.py --model ./qwen3-vl-4b-instruct --train_json ./coco_2014_caption/train.json --output_dir ./output/Qwen3-VL-4Blora
Remember to run `uv run python -c "import swanlab; swanlab.login(api_key='...')"` once if you plan to train with SwanLab.
EOF

