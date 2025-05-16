import torchvision
import torchvision.transforms as transforms
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from torchvision.models import VGG19_BN_Weights

# 1. 加载预训练 VGG19_BN，并只取前 30 层（到 relu4_1）
def get_encoder():
  vgg = torchvision.models.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
  return vgg


if __name__=='main':
  encoder = torchvision.models.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT).features[:30].eval()  # 前30层包含 relu4_1
  # 2. 准备 TensorBoard writer
  writer = SummaryWriter("./test_logs_vgg30")

  # 3. 加载并处理原始图像（用于对比）
  in_img_path = "./goat.jpg"
  pil_img = Image.open(in_img_path).convert("RGB")

  # 4a. 用于可视化的 transform（仅 Resize + ToTensor，保证 0~1）
  to_tensor = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor()
  ])
  orig_tensor = to_tensor(pil_img)  # [3, 224, 224]

  # 4b. 用于送入 encoder 的预处理（Resize → ToTensor → Normalize）
  preprocess = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
  ])
  input_tensor = preprocess(pil_img).unsqueeze(0)  # [1, 3, 224, 224]

  # 5. 前向提取特征
  with torch.no_grad():
      feat = encoder(input_tensor)  # [1, 512, 28, 28]  # 经3次池化后空间尺寸为28×28
  print(feat.size())
  # 6. 准备特征图：取 batch 0，unsqueeze 出通道维度
  features = feat[0].cpu().detach().unsqueeze(1)  # [512, 1, 28, 28]

  # 7. 拼成一张大图（32 列 × 16 行，可容纳512个通道）
  #    → nrow=32 列, 会得到 grid of size [1, 16*28, 32*28] = [1, 448, 896]
  feature_grid = vutils.make_grid(
      features,
      nrow=32,
      normalize=True,
      scale_each=True
  )  # 输出 shape: [1, 448, 896]

  # 8. 写入 TensorBoard
  writer.add_image("Original Goat (224*224)", orig_tensor)
  writer.add_image("VGG19_BN relu4_1 Feature Maps (512 ch, 28*28)", feature_grid)
  writer.close()

  # 9. 在终端运行：
  #    tensorboard --logdir=./test_logs_vgg21
  # 然后浏览器打开 http://localhost:6006/ 即可同时看到：
  # - “Original Goat (224×224)”：原始图像
  # - “VGG19_BN relu4_1 Feature Maps (512 ch, 28×28)”：512个通道的网格

''    '''vgg.structures={VGG(
    (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)

    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU(inplace=True)
    (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (12): ReLU(inplace=True)

    (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): ReLU(inplace=True)
    (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (19): ReLU(inplace=True)
    (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (22): ReLU(inplace=True)
    (23): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (24): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (25): ReLU(inplace=True)

    (26): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    (27): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (29): ReLU(inplace=True)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (32): ReLU(inplace=True)
    (33): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (34): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (35): ReLU(inplace=True)
    (36): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (37): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (38): ReLU(inplace=True)

    (39): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (42): ReLU(inplace=True)
    (43): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (44): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (45): ReLU(inplace=True)
    (46): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (47): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (48): ReLU(inplace=True)
    (49): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (50): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (51): ReLU(inplace=True)

    (52): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)}'''