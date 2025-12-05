# 配置文件使用说明

## 概述

项目使用 YAML 配置文件来管理所有训练参数，参考 LLaMAFactory 的设计风格。所有配置集中在 `config.yaml` 文件中，方便管理和修改。

## 使用方法

### 基本使用

```bash
# 使用默认配置文件 config.yaml
python MoeLORA.py

# 指定自定义配置文件
python MoeLORA.py --config my_config.yaml
```

### 命令行参数覆盖

项目支持通过命令行参数覆盖配置文件中的值，参考 LLaMAFactory 的实现风格。命令行参数的**优先级高于配置文件**。

```bash
# 使用配置文件，并通过命令行参数覆盖部分配置
python MoeLORA.py \
  --model ./qwen3-vl-4b-instruct \
  --train_json ./coco_2014_caption/train.json \
  --output_dir ./output/Qwen3-VL-4Blora

# 或者只覆盖部分参数
python MoeLORA.py --model ./other-model
```

**支持的命令行参数**：

| 参数 | 说明 | 覆盖的配置项 |
|------|------|-------------|
| `--config` | 配置文件路径（默认：`config.yaml`） | - |
| `--model` | 模型路径 | `model.model_name_or_path` |
| `--train_json` | 训练数据 JSON 文件路径 | `dataset.train_json_path` |
| `--output_dir` | 输出目录 | `training.output_dir` |

**使用场景**：
- 快速测试不同模型或数据集，无需修改配置文件
- 在 CI/CD 流程中动态指定参数
- 临时覆盖配置进行实验

## 配置项说明

### 1. 模型配置 (`model`)

```yaml
model:
  model_name_or_path: "./qwen3-vl-4b-instruct"  # 模型路径
  trust_remote_code: true                        # 是否信任远程代码
  use_cache: false                               # 训练时关闭缓存
  gradient_checkpointing: true                    # 启用梯度检查点
```

### 2. 数据配置 (`dataset`)

```yaml
dataset:
  train_json_path: "data_vl.json"               # 训练数据路径
  max_train_samples: null                        # 最大训练样本数（null=全部）
  test_samples: 4                                # 测试集样本数
  max_length: 8192                               # 最大序列长度
  image_resize_height: 280                       # 图像高度
  image_resize_width: 280                        # 图像宽度
```

**重要**：`max_train_samples` 用于限制训练数据量：
- `null` 或 `-1`：使用全部数据
- 数字：只使用前 N 条数据（用于快速测试）

### 3. 训练配置 (`training`)

```yaml
training:
  output_dir: "./output/Qwen3-VL-4Bmoelora"      # 输出目录
  per_device_train_batch_size: 1                 # 批次大小
  gradient_accumulation_steps: 8                 # 梯度累积步数
  num_train_epochs: 5                            # 训练轮数
  max_steps: null                                # 最大步数（覆盖epochs）
  learning_rate: 1.0e-4                          # 学习率
  fp16: true                                     # 使用FP16
  dataloader_num_workers: 0                      # Windows建议为0
```

### 4. LoRA/MoeLoRA 配置 (`lora`)

```yaml
lora:
  lora_type: "moelora"                           # "lora" 或 "moelora"
  r: 8                                            # LoRA 秩
  lora_alpha: 16                                  # LoRA alpha
  lora_dropout: 0.1                              # LoRA dropout
  num_experts: 2                                 # MoeLoRA 专家数
  gate_dropout: 0.1                              # 门控 dropout
  target_modules:                                # 目标模块列表
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
```

### 5. 量化配置 (`quantization`)

```yaml
quantization:
  load_in_4bit: true                             # 启用4-bit量化
  bnb_4bit_use_double_quant: true                # 双量化
  bnb_4bit_quant_type: "nf4"                     # 量化类型
  bnb_4bit_compute_dtype: "float16"              # 计算精度
```

### 6. SwanLab 配置 (`swanlab`)

```yaml
swanlab:
  enabled: true                                  # 是否启用
  api_key: "your_api_key_here"                   # API Key（可直接在配置文件中设置，推荐）
  project: "Qwen3-VL-finetune"                   # 项目名称
  experiment_name: "qwen3-vl-coco2014"           # 实验名称
```

**SwanLab API Key 设置方式**：

在 `config.yaml` 中设置 `api_key`：
```yaml
swanlab:
  api_key: "your_api_key_here"  # 设置为你的 API Key
```

如果配置文件中 `api_key` 为 `null` 或未设置，SwanLab 会提示交互式输入（Windows 上可能出现 grep 错误）。

## 配置示例

### 快速测试配置（少量数据）

```yaml
dataset:
  max_train_samples: 10  # 只训练10条数据
  test_samples: 2

training:
  num_train_epochs: 1
  save_steps: 10
```

### 完整训练配置

```yaml
dataset:
  max_train_samples: null  # 使用全部数据

training:
  num_train_epochs: 5
  save_steps: 100
```

### 使用标准 LoRA（非 MoeLoRA）

```yaml
lora:
  lora_type: "lora"  # 改为 "lora"
  # num_experts 和 gate_dropout 会被忽略
```

## 常见问题

### Q: 如何快速测试训练流程？

A: 在 `config.yaml` 中设置：
```yaml
dataset:
  max_train_samples: 5  # 只训练5条数据
training:
  num_train_epochs: 1
  save_steps: 1
```

### Q: 如何禁用 SwanLab？

A: 设置：
```yaml
swanlab:
  enabled: false
```

### Q: 如何修改模型路径？

A: 修改：
```yaml
model:
  model_name_or_path: "/path/to/your/model"
```

### Q: 如何调整训练数据量？

A: 有两种方式：

**方式一**：修改配置文件
```yaml
dataset:
  max_train_samples: 100  # 只使用前100条数据
```

**方式二**：使用命令行参数（如果支持该参数）
```bash
python MoeLORA.py --train_json ./other_data.json
```

### Q: 命令行参数和配置文件哪个优先级更高？

A: **命令行参数优先级高于配置文件**。如果同时指定，命令行参数会覆盖配置文件中的对应值。

例如：
- 配置文件中：`model.model_name_or_path: "./qwen3-vl-4b-instruct"`
- 命令行指定：`--model ./other-model`
- 最终使用：`./other-model`

## 配置文件使用

```bash
# 使用默认配置文件 config.yaml
python MoeLORA.py

# 使用自定义配置文件
python MoeLORA.py --config my_config.yaml

# 使用配置文件 + 命令行参数覆盖
python MoeLORA.py --config config.yaml --model ./custom-model
```

**注意**：
- SwanLab API Key 可以通过环境变量 `SWANLAB_API_KEY` 设置，避免在配置文件中硬编码
- 命令行参数实现参考 LLaMAFactory 风格，使用点号路径映射到配置文件

