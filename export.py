# export.py
import torch
from src.model import StyleTransferNet

device = 'cuda'  # C++ 一般在 CPU 上测试，也可以改成 'cuda'
# 1) 实例化并加载训练好的权重
model = StyleTransferNet(decoder_ckpt="decoder.pth", device=device)
model.eval()

# 2) 构造示例输入（batch_size=1, 3x224x224）
example_content = torch.randn(1, 3, 224, 224, device=device)
example_style   = torch.randn(1, 3, 224, 224, device=device)

# 3) 通过 tracing 生成 TorchScript
traced_script = torch.jit.trace(model, (example_content, example_style))

# 4) 保存为 .pt
traced_script.save("style_transfer.pt")
print("TorchScript 模型已保存为 style_transfer.pt")
