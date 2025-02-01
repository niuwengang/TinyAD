import torch

# 检查 PyTorch 版本
print("PyTorch version:", torch.__version__)

# 检查是否支持 GPU
print("CUDA available:", torch.cuda.is_available())

# 如果支持 GPU，打印当前设备信息
if torch.cuda.is_available():
    print("Current device:", torch.cuda.get_device_name(0))