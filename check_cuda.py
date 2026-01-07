import torch
import sys

print("=== CUDA诊断信息 ===")
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA不可用，可能原因:")
    print("1. NVIDIA驱动未安装或版本过旧")
    print("2. CUDA工具包未安装")
    print("3. PyTorch版本与CUDA版本不匹配")

# 测试简单的张量操作
print("\n=== 张量操作测试 ===")
try:
    x = torch.tensor([1.0, 2.0, 3.0])
    if torch.cuda.is_available():
        x = x.cuda()
        print(f"张量在GPU上: {x.device}")
    else:
        print(f"张量在CPU上: {x.device}")
except Exception as e:
    print(f"错误: {e}")