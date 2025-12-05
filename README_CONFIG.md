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
  api_key: "your_api_key_here"                   # API Key（可直接在配置文件中设置）
  project: "Qwen3-VL-finetune"                   # 项目名称
  experiment_name: "qwen3-vl-coco2014"           # 实验名称
```

**SwanLab API Key 设置方式**（按优先级从高到低）：
1. **配置文件**：在 `config.yaml` 中设置 `api_key: "your_key_here"`
2. **环境变量**：`export SWANLAB_API_KEY="your_key_here"`（配置文件为 null 时使用）
3. **交互式输入**：如果两者都没有，SwanLab 会提示交互式输入（Windows 上可能出现 grep 错误）

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

A: 修改：
```yaml
dataset:
  max_train_samples: 100  # 只使用前100条数据
```

## 配置文件使用

```bash
# 使用默认配置文件 config.yaml
python MoeLORA.py

# 使用自定义配置文件
python MoeLORA.py --config my_config.yaml
```

**注意**：SwanLab API Key 可以通过环境变量 `SWANLAB_API_KEY` 设置，避免在配置文件中硬编码。

