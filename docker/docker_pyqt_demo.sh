#!/bin/bash

# Docker PyQt 演示脚本
# 用于在 Docker 容器中运行 PyQt5 图形界面演示

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

# 检查 X11 显示
if [ -z "$DISPLAY" ]; then
    echo "警告: DISPLAY 环境变量未设置，PyQt 应用可能无法显示"
    echo "请确保已设置 X11 转发或使用 VNC"
fi

# 运行容器并执行 PyQt 演示
echo "启动 PyQt 演示..."
docker run -it --rm \
    --gpus all \
    --name qwen3-vl-pyqt-demo \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$PROJECT_ROOT/qwen3-vl-4b-instruct:/app/qwen3-vl-4b-instruct" \
    -v "$PROJECT_ROOT/output:/app/output" \
    -v "$PROJECT_ROOT/coco_2014_caption:/app/coco_2014_caption" \
    -v "$PROJECT_ROOT/rag_data:/app/rag_data" \
    -v "$PROJECT_ROOT/multi_agent/knowledge_base:/app/multi_agent/knowledge_base" \
    -v "$PROJECT_ROOT/multi_agent/output:/app/multi_agent/output" \
    -v "$PROJECT_ROOT/image:/app/image" \
    -e CUDA_VISIBLE_DEVICES=0 \
    "$IMAGE_NAME" \
    python main_app_ui.py

