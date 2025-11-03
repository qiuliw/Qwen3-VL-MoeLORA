import torch
print("CUDA是否可用：", torch.cuda.is_available())  # 若返回False，说明是CPU版PyTorch
print("PyTorch版本：", torch.__version__)