import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoModelForImageTextToText,
    AutoProcessor,
)
import os
from glob import glob
import swanlab
import json
import yaml
import argparse
from transformers import BitsAndBytesConfig
from dataclasses import dataclass, field
from peft import PeftModel, inject_adapter_in_model
from peft.tuners.lora import LoraLayer

# 解析命令行参数
parser = argparse.ArgumentParser(description='Train Qwen3-VL with MoeLoRA')
parser.add_argument('--config', type=str, default='config.yaml',
                    help='Path to configuration file (default: config.yaml)')
args_cmd = parser.parse_args()

# 加载配置文件
def load_config(config_path: str):
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# 加载配置
config = load_config(args_cmd.config)
def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = config['dataset']['max_length']
    input_ids, attention_mask, labels = [], [], []
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  # 获取图像路径
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                    "resized_height": config['dataset']['image_resize_height'],
                    "resized_width": config['dataset']['image_resize_width'],
                },
                {"type": "text", "text": "COCO Yes:"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取文本
    image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()} #tensor -> list,为了方便拼接
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)


    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  #由（1,h,w)变换为（h,w）
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}


def predict(messages, model):
    # 准备推理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

# 4-bit量化配置（显存占用更低）
quant_cfg = config['quantization']
compute_dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
bnb_config = BitsAndBytesConfig(
    load_in_4bit=quant_cfg['load_in_4bit'],
    bnb_4bit_use_double_quant=quant_cfg['bnb_4bit_use_double_quant'],
    bnb_4bit_quant_type=quant_cfg['bnb_4bit_quant_type'],
    bnb_4bit_compute_dtype=compute_dtype_map.get(quant_cfg['bnb_4bit_compute_dtype'], torch.float16)
) if quant_cfg['load_in_4bit'] else None

# 模型路径
model_id = config['model']['model_name_or_path']
# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=config['model']['trust_remote_code'])
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=config['model']['trust_remote_code'])

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=config['model']['trust_remote_code']
)
model.config.use_cache = config['model']['use_cache']
if config['model']['gradient_checkpointing']:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 处理数据集：读取json文件
train_json_path = config['dataset']['train_json_path']
with open(train_json_path, 'r') as f:
    data = json.load(f)
    
# 限制训练数据数量
max_train_samples = config['dataset']['max_train_samples']
if max_train_samples and max_train_samples > 0:
    data = data[:max_train_samples]
    
# 拆分成训练集和测试集
test_samples = config['dataset']['test_samples']
train_data = data[:-test_samples] if len(data) > test_samples else data
test_data = data[-test_samples:] if len(data) > test_samples else []

with open("data_vl_train.json", "w") as f:
    json.dump(train_data, f)

with open("data_vl_test.json", "w") as f:
    json.dump(test_data, f)

train_ds = Dataset.from_json("data_vl_train.json")
train_dataset = train_ds.map(process_func)


import torch.nn as nn
@dataclass
class MOELoraConfig(LoraConfig):
    num_experts: int = field(default=2, metadata={"help": "专家模块数量（8G显存建议≤2）"})
    gate_dropout: float = field(default=0.1, metadata={"help": "路由门控的dropout"})
    expert_capacity: int = field(default=1, metadata={"help": "每个token最多激活的专家数"})
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    # 新增PEFT检查所需的属性（显式标记为非prompt learning）
    @property
    def is_prompt_learning(self):
        return False  # MOELora不属于prompt learning
    
    @property
    def is_adaption_prompt(self):
        return False

# 配置LoRA参数
lora_cfg = config['lora']
if lora_cfg['lora_type'] == 'moelora':
    peft_config = MOELoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg['target_modules'],
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['lora_alpha'],
        lora_dropout=lora_cfg['lora_dropout'],
        num_experts=lora_cfg['num_experts'],
        gate_dropout=lora_cfg['gate_dropout'],
        expert_capacity=1,
        bias="none",
    )
else:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg['target_modules'],
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['lora_alpha'],
        lora_dropout=lora_cfg['lora_dropout'],
        bias="none",
    )

#自定义 MOELora 层  每个专家是一组 LoRA 低秩矩阵（保持参数少的优势）；
class MOELoraLayer(LoraLayer):
    def __init__(self, base_layer, config):
        super().__init__(base_layer, config)
        # 补充必要属性（从基础层获取）
        self.in_features = base_layer.in_features  # 输入特征维度
        self.out_features = base_layer.out_features  # 输出特征维度
        self.dropout = config.lora_dropout  # dropout参数

        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.scaling = config.lora_alpha / config.r  # LoRA缩放因子（与传统LoRA一致
        # 为每个目标模块创建多个专家LoRA层（每个专家是一组低秩矩阵）
        self.experts = nn.ModuleList([
            self._create_lora_expert() for _ in range(self.num_experts)
        ])
        # 路由门控（输入：隐藏层特征，输出：专家权重）
        self.gate = nn.Linear(base_layer.out_features, self.num_experts)
        self.gate_dropout = nn.Dropout(config.gate_dropout)
    def _create_lora_expert(self):
        # 单个专家的LoRA结构（与原LoRA一致）
        return nn.Sequential(
            nn.Linear(self.in_features, self.r, bias=False),
            nn.Dropout(self.dropout),
            nn.Linear(self.r, self.out_features, bias=False)
        )  
      
    def forward(self, x):
        # 1. 基础层输出
        base_out = self.base_layer(x)
        # 2. 门控路由（计算每个专家的权重）
        gate_weights = torch.softmax(self.gate(x), dim=-1)  # (batch, seq_len, num_experts)
        gate_weights = self.gate_dropout(gate_weights)  # 防止门控过拟合
        # 3. 选择top-k专家（这里k=expert_capacity=1）
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.expert_capacity, dim=-1)
        # 4. 专家输出加权求和
        expert_out = torch.zeros_like(base_out)
        for i in range(self.num_experts):
            # 筛选激活当前专家的token
            mask = (top_k_indices == i).float().sum(dim=-1, keepdim=True)  # (batch, seq_len, 1)
            if mask.sum() == 0:
                continue  # 无token激活该专家，跳过
            # 专家计算
            expert_pred = self.experts[i](x)
            # 加权累加（乘以门控权重）
            expert_out += mask * (expert_pred * top_k_weights[:, :, i:i+1])
        # 5. 最终输出 = 基础输出 + 专家输出 * 缩放因子
        return base_out + expert_out * self.scaling

# 替换原get_peft_model，注入MOELora层
# 替换get_moe_peft_model函数为：
def get_moe_peft_model(model, peft_config):
    # 显式指定适配器层为MOELoraLayer
    peft_config.adapter_layer = MOELoraLayer
    return get_peft_model(model, peft_config)

#获取Lora模型
if lora_cfg['lora_type'] == 'moelora':
    peft_model = get_moe_peft_model(model, peft_config)
else:
    peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
# 配置训练参数
train_cfg = config['training']
args = TrainingArguments(
    output_dir=train_cfg['output_dir'],
    per_device_train_batch_size=train_cfg['per_device_train_batch_size'],
    gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],
    logging_steps=train_cfg['logging_steps'],
    logging_first_step=train_cfg['logging_first_step'],
    num_train_epochs=train_cfg['num_train_epochs'] if train_cfg['max_steps'] is None else None,
    max_steps=train_cfg['max_steps'],
    save_steps=train_cfg['save_steps'],
    learning_rate=train_cfg['learning_rate'],
    optim=train_cfg.get('optim', 'adamw_torch'),
    fp16=train_cfg['fp16'],
    bf16=train_cfg.get('bf16', False),
    save_on_each_node=train_cfg['save_on_each_node'],
    gradient_checkpointing=train_cfg['gradient_checkpointing'],
    dataloader_num_workers=train_cfg.get('dataloader_num_workers', 0),
    report_to=config['misc']['report_to'],
    seed=config['misc'].get('seed', 42),
)

# 设置SwanLab回调
swanlab_callback = None
if config['swanlab']['enabled']:
    # 如果配置文件中有 api_key，提前初始化避免交互式登录
    swanlab_api_key = config['swanlab'].get('api_key')
    if swanlab_api_key:
        swanlab.init(
            project=config['swanlab']['project'],
            experiment_name=config['swanlab']['experiment_name'],
            api_key=swanlab_api_key,
        )
    
    swanlab_config = config['swanlab']['config'].copy()
    swanlab_config.update({
        "train_data_number": len(train_data),
        "lora_rank": lora_cfg['r'],
        "lora_alpha": lora_cfg['lora_alpha'],
        "lora_dropout": lora_cfg['lora_dropout'],
        "lora_type": lora_cfg['lora_type'],
    })
    swanlab_callback = SwanLabCallback(
        project=config['swanlab']['project'],
        experiment_name=config['swanlab']['experiment_name'],
        config=swanlab_config,
    )

# 配置Trainer
callbacks_list = [swanlab_callback] if swanlab_callback else []
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=callbacks_list,
)

# 开启模型训练
trainer.train()

# ====================测试模式===================
# 配置测试参数（与训练时的LoRA配置完全一致）
if lora_cfg['lora_type'] == 'moelora':
    val_config = MOELoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg['target_modules'],
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['lora_alpha'],
        lora_dropout=lora_cfg['lora_dropout'],
        num_experts=lora_cfg['num_experts'],
        gate_dropout=lora_cfg['gate_dropout'],
        expert_capacity=1,
        bias="none",
    )
else:
    val_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg['target_modules'],
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['lora_alpha'],
        lora_dropout=lora_cfg['lora_dropout'],
        bias="none",
    )

# 自动获取最新checkpoint
checkpoint_dirs = glob(f"{train_cfg['output_dir']}/checkpoint-*")
latest_checkpoint = max(checkpoint_dirs, key=os.path.getctime) if checkpoint_dirs else train_cfg['output_dir']
# 获取测试模型
val_peft_model = PeftModel.from_pretrained(model, model_id=latest_checkpoint, config=val_config)

# 读取测试数据
with open("data_vl_test.json", "r") as f:
    test_dataset = json.load(f)

test_image_list = []
for item in test_dataset:
    input_image_prompt = item["conversations"][0]["value"]
    # 去掉前后的<|vision_start|>和<|vision_end|>
    origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    
    messages = [{
        "role": "user", 
        "content": [
            {
            "type": "image", 
            "image": origin_image_path
            },
            {
            "type": "text",
            "text": "COCO Yes:"
            }
        ]}]
    
    response = predict(messages, val_peft_model)
    messages.append({"role": "assistant", "content": f"{response}"})
    print(messages[-1])

    test_image_list.append(swanlab.Image(origin_image_path, caption=response))

swanlab.log({"Prediction": test_image_list})

# 在Jupyter Notebook中运行时要停止SwanLab记录，需要调用swanlab.finish()
swanlab.finish()
