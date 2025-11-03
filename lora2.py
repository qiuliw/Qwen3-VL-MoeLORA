import os
import json
import io
import random
import numpy as np
import torch

from datasets import Dataset as HFDataset  
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForImageTextToText,
    AutoProcessor,
)
import swanlab
from transformers import BitsAndBytesConfig

# ========== 配置 ==========
MAX_DATA_NUMBER = 500
OUT_DIR = "./output/Qwen3-VL-4Blora"
os.makedirs(OUT_DIR, exist_ok=True)

# --------- 4-bit 量化配置（保持） ----------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# --------- 模型 / tokenizer / processor 路径 ----------
model_id = "./qwen3-vl-4b-instruct"

# --------- Tokenizer / Processor 初始化 ----------
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id)

# 确保 tokenizer 有 pad token（若无则使用 eos）
if tokenizer.pad_token_id is None:  # <<< MOD
    tokenizer.pad_token_id = tokenizer.eos_token_id  # <<< MOD

# --------- 加载量化模型到单卡 GPU（避免 device_map="auto" 带来的复杂性） ----------
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",     
    trust_remote_code=True
)
# 必要配置以支持 gradient checkpointing
model.config.use_cache = False  # <<< MOD: 训练时禁止 use_cache
try:
    model.gradient_checkpointing_enable()
except Exception:
    pass
model.enable_input_require_grads()  # 保持你原有调用

# ========= 数据准备 =========
train_json_path = "data_vl.json"
with open(train_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    train_data = data[:-4]
    test_data = data[-4:]

with open("data_vl_train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False)
with open("data_vl_test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False)

raw_ds = HFDataset.from_json("data_vl_train.json")

# ========== process_func (返回可序列化类型，不返回 torch.Tensor) ==========
def process_func(example):
    """
    返回字典：
      - input_ids (list[int])
      - attention_mask (list[int])
      - labels (list[int])           # 使用 -100 忽略前缀
      - pixel_values (ndarray)       # numpy.ndarray
      - image_grid_thw (ndarray)     # numpy.ndarray
    """
    MAX_LENGTH = 8192
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]

    # 提取图片路径
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]

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

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    proc_outputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # 将 token 部分转换为 Python list（便于序列化存入 Arrow 表）
    instr_ids = proc_outputs["input_ids"][0].tolist()
    instr_attn = proc_outputs["attention_mask"][0].tolist()

    resp = tokenizer(output_content, add_special_tokens=False)
    resp_ids = resp["input_ids"]
    resp_attn = resp.get("attention_mask", [1] * len(resp_ids))

    # 拼接 input 与 labels（labels 的 prompt 部分用 -100）
    input_ids = instr_ids + resp_ids
    attention_mask = instr_attn + resp_attn
    labels = [-100] * len(instr_ids) + resp_ids  # <<< MOD: 用 -100 忽略 prompt 区域

    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    # ---------- 关键改动：把视觉输出转为可序列化类型 ----------
    # pixel_values: 取第0张并转 numpy（float32）
    pixel_values = None
    if "pixel_values" in proc_outputs and proc_outputs["pixel_values"] is not None:
        pixel_values = proc_outputs["pixel_values"][0].cpu().numpy()

    # image_grid_thw: 将其转为 Python tuple of ints（非常关键）
    image_grid_thw = None
    if "image_grid_thw" in proc_outputs and proc_outputs["image_grid_thw"] is not None:
        arr = proc_outputs["image_grid_thw"][0].cpu().numpy()
        # 将任何形状扁平化为一维，再转为 int tuple，例如 (h, w) 或 (something,)
        flat = arr.flatten()
        image_grid_thw = tuple(int(x) for x in flat.tolist())

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

# ========== 处理所有样本（避免直接用 map 返回 tensors） ==========
# 仅处理前 MAX_DATA_NUMBER 个样本以节省资源
raw_list = list(raw_ds)[:MAX_DATA_NUMBER]
processed = [process_func(x) for x in raw_list]  # <<< MOD: 逐条处理，返回可序列化类型
train_dataset = HFDataset.from_list(processed)  # <<< MOD: 用 from_list 构造实际训练集

print("训练集样本数：", len(train_dataset))

# ========== 自定义 collate_fn（替换 DataCollatorForSeq2Seq） ==========
from typing import List, Dict

def collate_fn(batch):
    """
    batch: list of dicts returned by process_func
    - Keep image_grid_thw as list of tuples (python ints), not tensor.
    - pixel_values stacked as tensor.
    """
    input_ids = [b["input_ids"] for b in batch]
    attn = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]

    # pad input_ids / attention_mask using tokenizer
    padded = tokenizer.pad({"input_ids": input_ids, "attention_mask": attn}, padding=True, return_tensors="pt")
    max_len = padded["input_ids"].shape[1]

    # pad labels to max_len using -100
    labs = []
    for lab in labels:
        t = torch.tensor(lab, dtype=torch.long)
        if t.size(0) < max_len:
            pad = torch.full((max_len - t.size(0),), -100, dtype=torch.long)
            t = torch.cat([t, pad], dim=0)
        else:
            t = t[:max_len]
        labs.append(t)
    labels_tensor = torch.stack(labs)

    # stack pixel_values (numpy -> tensor)
    pv_list = []
    for b in batch:
        pv = b["pixel_values"]
        if pv is None:
            # 安全兜底：若没有 pixel_values，创建空 tensor（理论上不应该发生）
            pv = np.zeros((3, 224, 224), dtype=np.float32)
        pv_list.append(torch.tensor(pv, dtype=torch.float32))
    pixel_values_tensor = torch.stack(pv_list)  # (B, C, H, W)

    # 关键：image_grid_thw 保持为 Python list-of-tuples（例如 [(h,w), (h,w), ...]）
    grid_list = []
    for b in batch:
        g = b["image_grid_thw"]
        if g is None:
            # 如果为空，使用占位，例如 (1,1)
            grid_list.append((1,))
        else:
            # already a tuple of ints from process_func; 保证都是 int
            grid_list.append(tuple(int(x) for x in g))

    return {
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "labels": labels_tensor,
        "pixel_values": pixel_values_tensor,
        "image_grid_thw": grid_list,   # <<<< 传 Python list-of-tuples 给模型
    }

# ========== 配置 LoRA（为 8G 显存做轻量配置） ==========
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        # 视觉模块：按需加入（不要一次全开，保持精简）
        "visual_q_proj", "visual_k_proj",
    ],
    inference_mode=False,
    r=8,               # <<< MOD: r=8 更适合 8GB 显存
    lora_alpha=16,     # <<< MOD: alpha 与 r 匹配
    lora_dropout=0.05,
    bias="none",
)

peft_model = get_peft_model(model, config)
# 确认可训练参数（仅 LoRA）
try:
    peft_model.print_trainable_parameters()  # <<< MOD: 调试信息，便于确认
except Exception:
    pass

# ========== Trainer 配置（降低 batch_size，增加 accumulation） ==========
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,      # <<< MOD: 单卡 8G 建议 1
    gradient_accumulation_steps=8,      # <<< MOD: 保持有效 batch size
    logging_steps=10,
    logging_first_step=True,
    num_train_epochs=5,
    save_steps=500,                     # 减少过于频繁保存带来的 I/O
    learning_rate=1e-4,
    fp16=True,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

# ========== SwanLab 回调（保留你的配置，调整 lora_rank 同步） ==========
swanlab_callback = SwanLabCallback(
    project="Qwen3-VL-finetune",
    experiment_name="qwen3-vl-coco2014",
    config={
        "model": "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct",
        "dataset": "https://modelscope.cn/datasets/modelscope/coco_2014_caption/quickstart",
        "github": "https://github.com/datawhalechina/self-llm",
        "prompt": "COCO Yes: ",
        "train_data_number": len(train_data),
        "lora_rank": 8,            # <<< MOD: 与 config.r 保持一致
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    },
)

# ========== Trainer 初始化（使用自定义 collate_fn） ==========
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=collate_fn,    # <<< MOD: 使用自定义 collate_fn，替换 DataCollatorForSeq2Seq
    callbacks=[swanlab_callback],
)

# ========== 开始训练 ==========
trainer.train()

# ==================== 测试/推理（加载 LoRA checkpoint） ====================
# 加载 base 模型（与训练时量化/映射一致），再注入 LoRA checkpoint
base_for_infer = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": "cuda"},
    trust_remote_code=True
)
base_for_infer.config.use_cache = False

# 注意：给出 checkpoint 目录（你的保存路径）
checkpoint_dir = OUT_DIR  # 若你保存到了子目录，请改为具体检查点路径
infer_model = PeftModel.from_pretrained(base_for_infer, checkpoint_dir)  # <<< MOD: 正确加载方式
infer_model.to("cuda").eval()

# 读取测试数据并进行预测（与你原脚本保持一致）
with open("data_vl_test.json", "r", encoding="utf-8") as f:
    test_dataset = json.load(f)

test_image_list = []
for item in test_dataset:
    input_image_prompt = item["conversations"][0]["value"]
    origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": origin_image_path},
                {"type": "text", "text": "COCO Yes:"},
            ],
        }
    ]

    # 这里调用你原有的 predict 函数：注意 ensure inputs device on cuda
    def predict_local(messages, model_local):
        text_local = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs_local, video_inputs_local = process_vision_info(messages)
        inputs_local = processor(
            text=[text_local], images=image_inputs_local, videos=video_inputs_local, padding=True, return_tensors="pt"
        )
        # 把 inputs_local 转到 cuda
        inputs_local = {k: v.to("cuda") for k, v in inputs_local.items()}
        generated_ids = model_local.generate(**inputs_local, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_local["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    response = predict_local(messages, infer_model)
    messages.append({"role": "assistant", "content": f"{response}"})
    print(messages[-1])
    test_image_list.append(swanlab.Image(origin_image_path, caption=response))

swanlab.log({"Prediction": test_image_list})
swanlab.finish()
