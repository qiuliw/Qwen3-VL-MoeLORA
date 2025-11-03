# Qwen3-VL-MoeLORA

Qwen3-VL-4B-Instruct 模型的多模态微调过程，采用 MOELora（混合专家 LoRA） 技术，使用 COCO 2014 数据集（前 500/1000）

![](image/image.png)
将 MOE（混合专家模型）的动态路由机制与 LoRA 的低秩适配结合，在有限显存硬件上实现高效微调 —— 通过控制专家数量（2 个专家）和单专家低秩结构（r=8）

##### 代码包含完整流程代码

（注：不包含 Qwen3-VL-4B-Instruct 模型代码和权重，请自行下载）：
coco_2014_caption 数据集 [coco_2014_caption](https://modelscope.cn/datasets/modelscope/coco_2014_caption/quickstart)
Qwen3-VL-4B-Instruct 模型 [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
下载后直接放在项目所在目录即可
ModelScope 数据集加载->多模态数据预处理->lora\MOELora 微调配置->SwanLab 训练可视化及微调后模型推理

#### 1.本地部署推理

模型从 huggingface 下载到本地后，将 test.py 中的 model_id 换为本地路径，运行 test.py 文件

![](image/image-20251018215438537.png)

#### 2.基本 lora 微调

lora 配置

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

#### 训练运行（log）

| 指标                          | 值                                                          | 备注                                                                 |
| ----------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------- |
| 模型（base）                  | `qwen3-vl-4b-instruct`（本地路径 `./qwen3-vl-4b-instruct`） | 你以 4-bit + PEFT 微调该模型                                         |
| 数据集样本数（train）         | 496 examples                                                | 日志 `Generating train split: 496 examples`                          |
| 最大上下文长度（脚本）        | 8192 tokens                                                 | 脚本中 `MAX_LENGTH = 8192`（你可在面试中说明：为长上下文设计）       |
| 微调方法                      | LoRA (PEFT) + 4-bit quantization (bnb nf4)                  | 脚本中使用 `LoraConfig` 与 `BitsAndBytesConfig(load_in_4bit=True)`   |
| 注入的可训练参数（日志）      | 5,898,240 trainable params                                  | 日志：`trainable params: 5,898,240`                                  |
| 模型总参数量（日志）          | 4,443,714,048 全量参数                                      | 日志：`all params: 4,443,714,048`                                    |
| trainable 百分比（日志）      | ~0.1327%                                                    | 日志：`trainable%: 0.1327`                                           |
| 训练轮次 (epochs)             | 5.0 epochs                                                  | 日志结尾显示 `epoch: 5.0`                                            |
| 总训练步数（global steps）    | 310 steps                                                   | 日志最后 `310/310`                                                   |
| 每 epoch 步数                 | ~62 steps/epoch                                             | `310 / 5 = 62`，与样本/批次配置一致                                  |
| per_device_train_batch_size   | 1 (已在脚本优化为 1)                                        | 你之前为 8GB 卡设置为 1（并用 gradient accumulation 提升有效 batch） |
| gradient_accumulation_steps   | 8                                                           | 因此 **有效 batch size = 1 \* 8 = 8**                                |
| 学习率（初始）                | 1e-4                                                        | `learning_rate=1e-4`                                                 |
| 学习率（训练末期）            | ≈3.23e-07                                                   | 训练日志显示最后 step 的 lr: `3.2258e-07`（线性/调度衰减）           |
| 训练总时长                    | 5108.6867 s ≈ 85.15 min                                     | 日志 `train_runtime: 5108.6867`                                      |
| 平均 train_loss（全程）       | ~1.70645                                                    | 日志 `train_loss: 1.7064486826619794`                                |
| 初始 batch loss（第一条日志） | 4.8942                                                      | 首条记录 `{'loss': 4.8942, ...}`                                     |
| 训练样本吞吐                  | 0.485 samples/s                                             | 日志 `train_samples_per_second: 0.485`                               |
| 训练步吞吐                    | 0.061 steps/s                                               | 日志 `train_steps_per_second: 0.061`                                 |
| 梯度范数（观测范围）          | 约 1.25 — 3.75（观测）                                      | 日志多处 `grad_norm` 值；可答：在 ~1.2–3.8 范围内波动                |
| 量化方式                      | 4-bit NF4 双量化，compute_dtype=float16                     | `bnb_4bit_quant_type="nf4"` + `bnb_4bit_compute_dtype=torch.float16` |
| mixed-precision               | fp16=True（Trainer）                                        | 脚本里 `fp16=True`                                                   |
| checkpoint 信息               | 至少保存到 `./output/Qwen3-VL-4Blora`（存在 checkpoint-62） | 日志后段尝试加载 `checkpoint-62` 出现路径校验警告（见下）            |
| 训练中已记录（监控）          | SwanLab（logs/可视化）                                      | swanlab 同步并上报了 run（见日志链接）                               |

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

注意：不要把本地路径以 `model_id=` 形式传给 `from_pretrained` 里会触发 HF repo id 验证（日志里已见错误提示）。直接把本地 checkpoint 目录路径作为第一个参数传入 `PeftModel.from_pretrained` 即可。

微调后推理结果
![](image/PixPin_2025-11-03_17-12-57.png)

###### 致谢:

[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
