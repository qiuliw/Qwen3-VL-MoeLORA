from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
import torch
from peft import PeftModel  # 新增PeftModel导入
# 1. 模型路径和配置（替换为你的本地基础模型路径）
model_id = "./qwen3-vl-4b-instruct"  # 例如："D:/models/qwen3-vl-4b-instruct"

# 4-bit量化配置（显存占用更低）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 改为4-bit量化
    bnb_4bit_use_double_quant=True,  # 双量化优化
    bnb_4bit_quant_type="nf4",  # 推荐的4-bit类型
    bnb_4bit_compute_dtype=torch.float16  # 计算时用float16加速
)

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",  # 此时显存足够，会完全加载到GPU
    dtype=torch.float16,
    trust_remote_code=True
)

# 2. 加载LoRA微调权重（替换为你的实际checkpoint路径，如最新的checkpoint-310）
lora_checkpoint = "output/Qwen3-VL-4Blora/checkpoint-310"  
model = PeftModel.from_pretrained(model, lora_checkpoint)
model = model.to(model.device)  # 确保模型在GPU上

# 3. 加载处理器（处理图像和文本的统一工具）
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    padding_side="left"  # 左对齐padding，避免生成错误
)

# 4. 准备对话内容（图文混合或纯文本）
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",  # 示例图片
            },
            {"type": "text", "text": "用中文描述这张图片"}  # 你的问题
        ],
    }
]

# 5. 预处理输入（将图文转换为模型可识别的格式）
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # 自动添加"助手："等提示
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)  # 移到模型所在设备（GPU）

# 6. 生成回答（控制长度，避免显存溢出）
with torch.no_grad():  # 关闭梯度计算，节省显存
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,  # 生成的最大token数（8G显存建议≤300）
        temperature=0.7,  # 随机性（0-1）
        do_sample=True
    )

# 7. 解析输出（只保留生成的回答部分）
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print("模型回答：")
print(output_text)