# Qwen3-VL-MoeLoRA

![](image/image.png)

> 在千问最新的多模态 image-text 模型 **Qwen3-VL-4B-Instruct** 上完成 MOELoRA（混合专家 LoRA）微调，同时打通 COCO 2014 数据处理、SwanLab 监控、LangChain + RAG + Qt 多智能体模型部署的全过程。
> 首先简单介绍一下 Qwen-vl 和 Qwen3-vl

<details>
<summary><strong>Qwen-VL</strong></summary>

> [Qwen3-VL Technical Report](https://arxiv.org/pdf/2511.21631)

如下图，Qwen-VL 系列 的训练过程分为三个阶段：

- **Stage1 为预训练**，目标是使用大量的图文 Pair 对数据对齐视觉模块和 LLM 的特征，这个阶段冻结 LLM 模块的参数；
- **Stage2 为多任务预训练**，使用更高质量的图文多任务数据（主要来源自开源 VL 任务，部分自建数据集），更高的图片像素输入，全参数训练；
- **Stage3 为指令微调阶段**，这个阶段冻结视觉 Encoder 模块，使用的数据主要来自大模型 Self-Instruction 方式自动生成，目标是提升模型的指令遵循和多轮对话能力。

![](image/Qwen-vl.png)
而最新开源的 Qwen3-vl 主要有如下创新：

- **Interleaved-MRoPE：** 在 时间/宽/高多维度做全频率分配的位置编码，提升长视频时序推理。
- **DeepStack：** 融合多层 ViT 视觉特征，强化细粒度对齐与识别。
- **Text–Timestamp Alignment：** 从 T-RoPE 走向“文本-时间戳”精准对齐，利于事件级视频定位。
  ![](image/qwen3-vl.jpg)

</details>

---

## 🚀 项目介绍与模块概览

1. **数据 → 训练 → 推理全链路脚本齐全**：`download_data2csv.py` 负责拉取 & 清洗 COCO Caption，`csv2json.py` 适配 Qwen3-VL 格式，`MoeLORA.py/ lora.py` 处理 LoRA / MoeLoRA 训练，`test.py` 快速验证，`multi_agent/` 则提供 LangChain + RAG + Qt 多智能体部署。
2. **本地化 RAG + 多模态多智能体助手**：PyQt5 桌面端 UI，整合 LangChain 检索、FAISS 向量库与本地 Qwen3-VL 推理，支持文本/图像问答、一键开关知识库引用。

- Advanced RAG：BM25 + FAISS 混合召回，结合 `BAAI/bge-reranker-base` Cross-Encoder 重排序，默认输出最相关的 Top-N 片段。
- Multi-Agent 升级：新增 Reviewer 自检步骤（最多重写一次答案）以及 MCP 风格 `save_session_summary` 工具，可把聊天记录一键导出 Markdown。

3. **轻量显存友好**：默认 4-bit NF4 量化 + LoRA，只需单卡 8G 也能跑完微调流程。
4. **SwanLab 全程可视化**：训练日志、指标可视化齐备，便于调优与复现。

项目提供 “数据下载 → 格式转换 →LoRA / MoeLoRA 训练 → 本地推理 → 多智能体部署” 的最小可复现工程，帮助你快速验证自定义知识库 + 多模态问答的完整闭环。

> 仓库不自带 Qwen3-VL-4B-Instruct 权重与 COCO 数据集，请按下文指引下载到指定目录。

## 📁 项目结构

```text
Qwen3-VL-MoeLORA/
├── coco_2014_caption/              # 数据集下载 & 转换产物
├── multi_agent/                    # LangChain + RAG + Qt 多智能体助手
│   ├── main_app.py                 # PyQt5 主程序
│   ├── langchain_wrappers.py       # 本地 Qwen 文本/多模态封装
│   ├── multi_agent.py              # Planner / Manager / Knowledge / Reviewer / Responder 编排
│   ├── model_client.py             # 模型加载与图片输入处理
│   ├── rag_pipeline.py             # 知识库加载 + 混合检索 + 交叉编码重排序
│   ├── requirements.txt            # 子模块依赖（PyQt5、LangChain、FAISS 等）
│   ├── tools.py                    # MCP 风格工具（save_session_summary 等）
│   └── knowledge_base/             # 默认知识库，可放 txt/md/pdf
├── download_model.py               # 下载 Qwen3-VL 基座到本地
├── download_data2csv.py            # 拉取 ModelScope 数据集
├── csv2json.py                     # 数据转换脚本
├── MoeLORA.py / lora.py            # LoRA / MoeLoRA 训练
├── test.py                         # 本地推理 DEMO
├── main_app_ui.py / main_langchain_ui.py # Qt UI 示例
├── output/                         # LoRA 训练结果
├── qwen3-vl-4b-instruct/           # 预期放置官方基座
└── requirements.txt                # 顶层依赖
```

## ⚙️ 快速上手

### 0. 环境要求：PyTorch 和 CUDA 版本对应

**⚠️ 重要提示：** 本项目需要 GPU 支持，请确保安装支持 CUDA 的 PyTorch 版本，而不是 CPU 版本。

#### 检查当前环境
```bash
# 检查 GPU 是否可用
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# 检查 NVIDIA 驱动和 CUDA 版本
nvidia-smi
```

#### PyTorch 和 CUDA 版本对应表

| PyTorch 版本 | CUDA 版本 | 安装命令 |
|-------------|----------|---------|
| 2.6.0 | CUDA 12.4 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| 2.5.0 | CUDA 12.4 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| 2.4.0 | CUDA 12.1 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| 2.3.0 | CUDA 12.1 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| 2.2.0 | CUDA 11.8 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| 2.1.0 | CUDA 11.8 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |

#### 安装步骤

1. **检查 NVIDIA 驱动版本**（通过 `nvidia-smi` 查看）
2. **根据驱动支持的 CUDA 版本选择对应的 PyTorch 版本**
3. **卸载 CPU 版本的 PyTorch**（如果已安装）：
   ```bash
   pip uninstall torch torchvision -y
   ```
4. **安装支持 CUDA 的 PyTorch**（以 CUDA 12.4 为例）：
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
5. **验证安装**：
   ```bash
   python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('✅ CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
   ```

> 💡 **提示**：如果 `torch.cuda.is_available()` 返回 `False`，请检查：
> - NVIDIA 驱动是否正确安装
> - PyTorch 是否安装了 CPU 版本（版本号中不包含 `+cu`）
> - CUDA 版本是否匹配

### 1. 克隆 & 创建虚拟环境

使用 [uv](https://github.com/astral-sh/uv) 快速创建 Python 3.12 环境：

**Windows:**
```powershell
# 安装 uv (如果未安装)
powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"

# 将 uv 添加到 PATH (当前会话)
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

# 方法2: 永久添加到 PATH (需要重启 PowerShell 终端才能生效)
# [Environment]::SetEnvironmentVariable("Path", "$env:USERPROFILE\.local\bin;$env:Path", "User")
# 然后关闭并重新打开 PowerShell 终端

# 验证 uv 是否可用
uv --version
# 如果显示版本号，说明配置成功

# 克隆项目
git clone https://github.com/<your-account>/Qwen3-VL-MoeLORA.git
cd Qwen3-VL-MoeLORA

# 使用 uv 创建 Python 3.12 虚拟环境并安装依赖
uv venv --python 3.12
.venv\Scripts\activate

# ⚠️ 重要：检查 PyTorch 和 CUDA 版本
# 确保安装支持 CUDA 的 PyTorch 版本，而不是 CPU 版本
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
# 如果 CUDA available 为 False，请参考下面的版本对应表重新安装
uv pip install -r requirements.txt
```

**Linux/Mac:**
```bash
# 安装 uv (如果未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆项目
git clone https://github.com/<your-account>/Qwen3-VL-MoeLORA.git
cd Qwen3-VL-MoeLORA

# 使用 uv 创建 Python 3.12 虚拟环境并安装依赖
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

> 多智能体桌面端依赖额外的 PyQt5/FAISS，可在 `multi_agent/` 目录执行 `uv pip install -r requirements.txt`。

### 2. 下载模型与数据集

```powershell
# 拉取官方基座
python download_model.py --target ./qwen3-vl-4b-instruct

# 下载 COCO Caption 示例并写入 CSV
python download_data2csv.py --output ./coco_2014_caption/train.csv

# 转换为 Qwen3-VL JSON（可指定条目）
python csv2json.py --csv ./coco_2014_caption/train.csv --json ./coco_2014_caption/train.json --top_k 500
```

#### 📝 自定义参数说明

| 参数 | 说明 | 默认值 | 修改位置 |
|------|------|--------|----------|
| **ModelScope Token** | 用于下载 ModelScope 数据集和模型的认证 token。如果下载失败，可在 [ModelScope](https://modelscope.cn) 注册账号并获取 token，然后设置环境变量：<br>`$env:MODELSCOPE_API_TOKEN="your_token"` (Windows)<br>`export MODELSCOPE_API_TOKEN="your_token"` (Linux/Mac) | 无 | 环境变量或 `download_data2csv.py` / `download_model.py` 中的 `token` 参数 |
| **SwanLab Token** | 用于训练数据可视化的认证 token。SwanLab 提供训练过程的可视化监控，包括损失曲线、学习率等指标。获取方式：<br>1. 访问 [SwanLab](https://swanlab.cn) 注册账号<br>2. 在个人设置中获取 API Key<br>3. 在 `config.yaml` 中设置 `swanlab.api_key: "your_api_key"` | 无 | `config.yaml` 配置文件中的 `swanlab.api_key` |

### 3. 快速推理（基座或 LoRA）

```powershell
python test.py --model ./qwen3-vl-4b-instruct --image ./image/demo.jpg --prompt "描述这张图片"
```

若已完成 LoRA 训练，可将 `--model` 指向合并后的权重或直接在 `test.py` 中加载 `PeftModel`。

### 4. 启动 LangChain + RAG + Qt 多智能体助手

```powershell
cd multi_agent
pip install -r requirements.txt  # 首次执行
python main_app.py
```

左侧输入问题/上传图片，右侧会展示 Planner/Manager 计划、RAG 检索摘要与最终回复。

#### 🌟 新版多智能体亮点

- **Advanced RAG**：BM25 + FAISS 融合召回，叠加 Cross-Encoder 重排序，显著降低“查不到/查不准”问题。
- **Reviewer QA Loop**：Responder 生成答案后，Reviewer 以 PASS/RETRY 形式复核；如不合格，会把建议反馈给 Responder 重写一次，提升回答完整性。
- **MCP 风格归档工具**：内置 `save_session_summary`，当聊天语句包含“总结对话 / 保存记录 / archive / summary …”时，会把完整对话、调度计划与检索证据写入 `multi_agent/output/reports/*.md`，回答里也会显示文件路径。
- **GUI 自动记录上下文**：前端持续跟踪每轮问答，生成总结时无需手工复制内容，Agent 会自动读取历史对话。

> 可以看到智能体正确回答我的问题：小苔藓于今年 9 月 25 日保送到厦门大学（也就是本人，哈哈，目前 GUI 还有点小问题，文字内容不长展示不全）

![](image/image1.png)

> 导出会话记录
> ![](image/images3.png)

### 5. LoRA / MoeLoRA 微调

项目支持两种配置方式：**YAML 配置文件**（推荐）和**命令行参数**（快速覆盖）。

#### 方式一：使用配置文件（推荐）

```powershell
# 使用默认配置文件 config.yaml
python MoeLORA.py

# 使用自定义配置文件
python MoeLORA.py --config my_config.yaml
```

所有参数在 `config.yaml` 中统一管理，参考 LLaMAFactory 风格。详细配置说明见 [README_CONFIG.md](README_CONFIG.md)。

#### 方式二：命令行参数覆盖

```powershell
# 使用配置文件，并通过命令行参数覆盖部分配置
python MoeLORA.py \
  --model ./qwen3-vl-4b-instruct \
  --train_json ./coco_2014_caption/train.json \
  --output_dir ./output/Qwen3-VL-4Blora
```

**命令行参数优先级高于配置文件**，可以快速覆盖常用参数而无需修改配置文件。

**支持的命令行参数**（覆盖同名配置，未列出的参数仍从 `config.yaml` 读取）：
- `--config`：配置文件路径（默认 `config.yaml`）
- `--model`：模型路径（覆盖 `model.model_name_or_path`）
- `--train_json`：训练数据 JSON 路径（覆盖 `dataset.train_json_path`）
- `--output_dir`：输出目录（覆盖 `training.output_dir`）
- 训练超参覆盖：`--per_device_train_batch_size`、`--gradient_accumulation_steps`、`--num_train_epochs`、`--save_steps`、`--logging_steps`、`--logging_first_step`、`--learning_rate`、`--fp16`

默认训练超参（与原脚本一致，详见 `config.yaml`）：
- `per_device_train_batch_size=1`，`gradient_accumulation_steps=8`
- `num_train_epochs=5`（未使用 `max_steps`）
- `save_steps=100`，`logging_steps=10`，`logging_first_step=5`
- `learning_rate=1e-4`
- `fp16=true`，`gradient_checkpointing=true`

脚本默认启用 BitsAndBytes 4-bit 与 PEFT，可根据显存情况调整 `r`、`lora_alpha`、`gradient_accumulation_steps` 等参数。训练完成后产物位于 `output/`，可被多智能体或 `test.py` 直接加载。

##### 代码包含完整流程代码

（注：不包含 Qwen3-VL-4B-Instruct 模型代码和权重，请自行下载）：
coco_2014_caption 数据集 [coco_2014_caption](https://modelscope.cn/datasets/modelscope/coco_2014_caption/quickstart)
Qwen3-VL-4B-Instruct 模型 [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
下载后直接放在项目所在目录即可（`./qwen3-vl-4b-instruct`）

ModelScope 数据集加载->多模态数据预处理->lora\MOELora 微调配置->SwanLab 训练可视化及微调后模型推理

## 🧠 LangChain + RAG + Qt 多智能体助手

- **Planner Agent**：拆解任务，列出 3-5 个关键步骤并提示是否需要知识库。
- **Manager Agent**：结合 Planner 输出与 RAG 预览，决定是否继续检索、给出额外提示。
- **Knowledge Agent**：基于 `rag_pipeline.py` 构建的 BM25 + FAISS 混合检索 + Cross-Encoder 重排序，支持 `.txt/.md/.pdf` 多编码加载。
- **Responder Agent**：通过 `model_client.py` 调用本地 Qwen3-VL（支持 LoRA 权重 & 图文输入），并能接收 Reviewer 的改写建议。
- **Reviewer Agent**：检查答案是否解决用户问题，必要时要求 Responder 重答。
- **Qt 前端**：`main_app.py` 复刻 `main_langchain_ui.py` 的交互体验，展示 Planner/Manager 面板、RAG 参考与最终答复

> 默认知识库目录：`multi_agent/knowledge_base/`，界面右下角可勾选 “启用知识库检索 (RAG)” 开关。若只需归档聊天，请输入“总结对话/保存记录”等指令，系统会自动调用 MCP 工具输出 Markdown。

##### 上传本地论文到知识库，对论文进行问答

![](image/image0.png)

#### 1.本地部署推理

模型从 huggingface 下载到本地后，将 test.py 中的 model_id 换为本地路径，运行 test.py 文件

![](image/image-20251018215438537.png)

#### 2.微调

lora 配置，见 MoeLORA.py 文件

```python
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    # 新增视觉编码器和交叉注意力层（Qwen3-VL特有模块）
    target_modules=[
        # 文本模块
        "q_proj", "k_proj", "v_proj", "o_proj"
        # 视觉模块
        "visual_q_proj", "visual_k_proj"],
    inference_mode=False,
    r=8,  # 8G显存建议r=16（原64可能显存不足）
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)
```

![](image/image-20251019172207276.png)

微调图像

![](image/image-20251019210718609.png)

<details>
<summary><strong>训练运行（log）</strong></summary>

| 指标                          | LoRA                                                        | MoeLoRA                                                    |
| ----------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------- |
| 模型（base）                  | `qwen3-vl-4b-instruct`（本地路径 `./qwen3-vl-4b-instruct`） | `qwen3-vl-4b-instruct`（`./qwen3-vl-4b-instruct`）         |
| 数据集样本数（train）         | 496 examples                                                | 496 examples                                               |
| 最大上下文长度（脚本）        | 8192 tokens                                                 | 8192 tokens                                                |
| 微调方法                      | LoRA (PEFT) + 4-bit quantization (bnb nf4)                  | MoeLoRA (多专家 LoRA) + 4-bit quantization (bnb nf4)       |
| 注入的可训练参数（日志）      | 5,898,240 trainable params                                  | 10,298,240 trainable params                                |
| 模型总参数量（日志）          | 4,443,714,048 全量参数                                      | 4,443,714,048 params                                       |
| trainable 百分比（日志）      | ~0.1327%                                                    | ≈ 0.245%                                                   |
| 训练轮次 (epochs)             | 5.0 epochs                                                  | 5.0 epochs                                                 |
| 总训练步数（global steps）    | 310 steps                                                   | 310 steps                                                  |
| 每 epoch 步数                 | ~62 steps/epoch                                             | 62                                                         |
| per_device_train_batch_size   | 1 (已在脚本优化为 1)                                        | 1                                                          |
| gradient_accumulation_steps   | 8                                                           | 8                                                          |
| 学习率（初始）                | 1e-4                                                        | 1e-4                                                       |
| 学习率（训练末期）            | ≈3.23e-07                                                   | 训练日志显示最后 step 的 lr: `3.2258e-07`（线性/调度衰减） |
| 训练总时长                    | 5108.6867 s ≈ 85.15 min                                     | 6595.6952≈109min                                           |
| 平均 train_loss（全程）       | ~1.70645                                                    | ~1.65432                                                   |
| 初始 batch loss（第一条日志） | 4.8942                                                      | 4.7856                                                     |
| 训练样本吞吐                  | 0.485 samples/s                                             | 0.44 samples/s                                             |
| 训练步吞吐                    | 0.061 steps/s                                               | 0.055 steps/s                                              |
| 梯度范数（观测范围）          | 约 1.25 — 3.75（观测）                                      | 1.5 — 4.5(波动)                                            |
| 量化方式                      | 4-bit NF4 双量化，compute_dtype=float16                     | 4-bit NF4 双量化，compute_dtype=float16                    |
| mixed-precision               | fp16=True（Trainer）                                        | `fp16=True`                                                |
| checkpoint 信息               | 保存到 `./output/Qwen3-VL-4Blora`                           | ./output/Qwen3-VL-4Bmoelora/checkpoint-\*                  |
| 训练中已记录（监控）          | SwanLab（logs/可视化）                                      | SwanLab（logs/可视化）                                     |

加载训练好的 LoRA checkpoint 做推理

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText

base = AutoModelForImageTextToText.from_pretrained(model_id,
                                                  quantization_config=bnb_config,
                                                  device_map={"": "cuda"},
                                                  trust_remote_code=True)
base.config.use_cache = False
infer_model = PeftModel.from_pretrained(base, "./output/Qwen3-VL-4Blora")  # 本地路径
infer_model.to("cuda").eval()

```

</details>

---

注意：不要把本地路径以 `model_id=` 形式传给 `from_pretrained` 里会触发 HF repo id 验证（日志里已见错误提示）。直接把本地 checkpoint 目录路径作为第一个参数传入 `PeftModel.from_pretrained` 即可。

微调后推理结果
![](image/PixPin_2025-11-03_17-12-57.png)

## 🧭 项目规划（Roadmap）

- [ ] **MCP (Model Context Protocol)**：将多智能体推理封装成 MCP Server，方便 IDE / Copilot Chat 直接调用。
- [ ] **知识库热更新**：支持远程向量数据库（Milvus/Elasticsearch）与在线文档自动同步。
- [ ] **评测自动化**：引入 LLM-as-a-judge、BLEU/ROUGE 等指标，对图文回答做自动验收。
- [ ] **数据扩展流水线**：抓取行业报告 →`csv2json.py` 自动转换 → 一键加入训练或 RAG。

###### 致谢:

[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
