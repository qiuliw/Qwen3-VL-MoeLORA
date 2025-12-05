FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3-pip git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip uv

WORKDIR /workspace
COPY requirements.txt ./requirements.txt
RUN uv pip install --system -r requirements.txt

# multi_agent has its own extras
COPY multi_agent/requirements.txt ./multi_agent/requirements.txt
RUN uv pip install --system -r multi_agent/requirements.txt

COPY . .

# runtime defaults
ENV TORCH_HOME=/workspace/.cache/torch \
    HF_HOME=/workspace/.cache/huggingface \
    ENABLE_CUDA=1

CMD ["/bin/bash"]

