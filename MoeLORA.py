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
from transformers import BitsAndBytesConfig
from dataclasses import dataclass, field
from peft import PeftModel, inject_adapter_in_model
from peft.tuners.lora import LoraLayer
def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 8192
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
                    "resized_height": 280,
                    "resized_width": 280,
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
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 改为4-bit量化
    bnb_4bit_use_double_quant=True,  # 双量化优化
    bnb_4bit_quant_type="nf4",  # 推荐的4-bit类型
    bnb_4bit_compute_dtype=torch.float16  # 计算时用float16加速
)
# 在modelscope上下载Qwen2-VL模型到本地目录下
model_id = "./qwen3-vl-4b-instruct" 
# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("qwen3-vl-4b-instruct", use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained("qwen3-vl-4b-instruct")

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto", # 此时显存足够，会完全加载到GPU
    dtype=torch.float16,
    trust_remote_code=True
)
model.config.use_cache = False              # <<< 确保训练时关闭 cache（避免警告）
model.gradient_checkpointing_enable()       # <<< 如果你想用 checkpointing 节省显存
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 处理数据集：读取json文件
# 拆分成训练集和测试集，保存为data_vl_train.json和data_vl_test.json
train_json_path = "data_vl.json"
with open(train_json_path, 'r') as f:
    data = json.load(f)
    train_data = data[:-4]
    test_data = data[-4:]

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
    # 新增PEFT检查所需的属性（显式标记为非prompt learning）
    @property
    def is_prompt_learning(self):
        return False  # MOELora不属于prompt learning
    
    @property
    def is_adaption_prompt(self):
        return False

#显卡资源有限时
config = MOELoraConfig(
    task_type=TaskType.CAUSAL_LM,
    # 目标模块保持不变（文本+视觉关键层）
    target_modules=[
        # 文本模块
        "q_proj", "k_proj", "v_proj", "o_proj",
        # 视觉模块
        "visual_q_proj", "visual_k_proj"
    ],
    inference_mode=False,
    r=8,  # 单个专家的秩（总参数=num_experts * r * ...，保持小秩适配显存）
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    # MOE特有参数（核心）
    num_experts=2,  # 专家数量（8G显存最多2个，避免显存爆炸）
    gate_dropout=0.1,  # 防止门控过拟合
    expert_capacity=1  # 每个token仅激活1个专家（减少计算量）
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
def get_moe_peft_model(model, config):
    # 显式指定适配器层为MOELoraLayer
    config.adapter_layer = MOELoraLayer
    return get_peft_model(model, config)
#获取Lora模型
peft_model = get_moe_peft_model(model, config)  # 使用MOELora模型
peft_model.print_trainable_parameters()  # 确认参数规模（应略大于原LoRA，2个专家约2倍）
# 配置训练参数
args = TrainingArguments(
    output_dir="./output/Qwen3-VL-4Bmoelora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    logging_first_step=5,
    num_train_epochs=5,
    save_steps=100,
    learning_rate=1e-4,
    fp16=True,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)
        
# 设置SwanLab回调
swanlab_callback = SwanLabCallback(
    project="Qwen3-VL-finetune",
    experiment_name="qwen3-vl-coco2014",
    config={
        "model": "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct",
        "dataset": "https://modelscope.cn/datasets/modelscope/coco_2014_caption/quickstart",
        "github": "https://github.com/datawhalechina/self-llm",
        "prompt": "COCO Yes: ",
        "train_data_number": len(train_data),
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    },
)

# 配置Trainer
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

# 开启模型训练
trainer.train()

# ====================测试模式===================
# 配置测试参数（与训练时的LoRA配置完全一致）
# 替换测试阶段的val_config
val_config = MOELoraConfig(  # 改为MOELoraConfig
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "visual_q_proj", "visual_k_proj"
    ],
    inference_mode=True,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    num_experts=2,  # 与训练时一致
    expert_capacity=1
)

# 自动获取最新checkpoint
checkpoint_dirs = glob("./output/Qwen3-VL-4Bmoelora/checkpoint-*")
latest_checkpoint = max(checkpoint_dirs, key=os.path.getctime) if checkpoint_dirs else "./output/Qwen3-VL-4Bmoelora"
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
