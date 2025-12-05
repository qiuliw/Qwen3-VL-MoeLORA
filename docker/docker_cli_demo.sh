#!/bin/bash

# Docker CLI 演示脚本
# 用于在 Docker 容器中运行命令行演示

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Docker 镜像名称
IMAGE_NAME="qwen3-vl-moelora:cu124"

# 检查镜像是否存在
if ! docker images | grep -q "qwen3-vl-moelora.*cu124"; then
    echo "构建 Docker 镜像..."
    cd "$PROJECT_ROOT"
    docker build -f docker/Dockerfile-cu121 -t "$IMAGE_NAME" .
fi

# 运行容器并执行 CLI 演示
echo "启动 CLI 演示..."
docker run -it --rm \
    --gpus all \
    --name qwen3-vl-cli-demo \
    -v "$PROJECT_ROOT/qwen3-vl-4b-instruct:/app/qwen3-vl-4b-instruct" \
    -v "$PROJECT_ROOT/output:/app/output" \
    -v "$PROJECT_ROOT/coco_2014_caption:/app/coco_2014_caption" \
    -v "$PROJECT_ROOT/rag_data:/app/rag_data" \
    -v "$PROJECT_ROOT/image:/app/image" \
    -e CUDA_VISIBLE_DEVICES=0 \
    "$IMAGE_NAME" \
    python test.py

