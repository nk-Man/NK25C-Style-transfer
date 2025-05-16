from src.model import StyleTransferNet
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch
from utils import content_loader


def unnormalize(x: torch.Tensor, device):
    # x: [1,3,224,224] 或 [B,3,224,224]
    mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
    return torch.clamp(x * std + mean, 0, 1)

if __name__=='__main__':
    # 1) 预加载 Style 图
    style_img = Image.open('pipipi.jpg').convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std =[0.229, 0.224, 0.225],
        ),
    ])
    style_tensor = preprocess(style_img).unsqueeze(0)

    # 2) 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = StyleTransferNet(decoder_ckpt='./decoder444.pth', device=device)
    model.eval()

    # 3) 打开 TensorBoard Writer
    writer = SummaryWriter(log_dir='show_logs')

    step = 0
    # 我们只想展示 100 张，batch_size=1 最简单
    for batch in content_loader:
        # batch: [B,3,224,224], 这里假设 batch_size=1 否则按需要拆分
        content_batch = batch.to(device)      # [1,3,224,224]
        with torch.no_grad():
            output = model(content_batch, style_tensor.to(device))  # [1,3,224,224]

        # 反归一化
        inp_vis  = unnormalize(content_batch, device)  # [1,3,224,224]
        out_vis  = unnormalize(output, device)

        # add_image 需要 [3, H, W]，去掉 batch 维
        writer.add_image("原始内容图", inp_vis[0].cpu(), global_step=step)
        writer.add_image("风格迁移后", out_vis[0].cpu(), global_step=step)

        step += 1
        if step >= 100:
            break

    # 4) 关闭 writer
    writer.close()
    print("已写入 100 张对比图到 TensorBoard，路径：show_logs/")
#tensorboard --logdir=show_logs