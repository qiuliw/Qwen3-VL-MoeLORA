import pandas as pd
import json
import os
from cli_utils import setup_argument_parser

# ==================== 命令行参数配置 ====================
CLI_ARGS_CONFIG = {
    'csv': {
        'type': str,
        'default': './coco-2024-dataset.csv',
        'help': 'Input CSV file path (default: ./coco-2024-dataset.csv)'
    },
    'json': {
        'type': str,
        'default': 'data_vl.json',
        'help': 'Output JSON file path (default: data_vl.json)'
    },
    'top_k': {
        'type': int,
        'default': None,
        'help': 'Only convert top K rows (default: all rows)'
    },
}

def main():
    parser = setup_argument_parser('Convert CSV to Qwen3-VL JSON format', CLI_ARGS_CONFIG)
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    
    # 载入CSV文件
    df = pd.read_csv(args.csv)
    
    # 限制行数
    if args.top_k and args.top_k > 0:
        df = df.head(args.top_k)
    
    conversations = []
    
    # 添加对话数据
    for i in range(len(df)):
        conversations.append({
            "id": f"identity_{i+1}",
            "conversations": [
                {
                    "from": "user",
                    "value": f"COCO Yes: <|vision_start|>{df.iloc[i]['image_path']}<|vision_end|>"
                },
                {
                    "from": "assistant", 
                    "value": df.iloc[i]['caption']
                }
            ]
        })
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存为Json
    with open(args.json, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully converted {len(conversations)} rows to {args.json}")

if __name__ == "__main__":
    main()
