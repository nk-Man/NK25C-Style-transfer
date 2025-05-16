# utils.py
from torchvision.datasets import Caltech101
import os
import torch.nn.functional as F
from torchvision.datasets import VOCDetection
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from pathlib import Path


# 超参数（可在 train.py 中调整）
CONTENT_WEIGHT = 2
STYLE_WEIGHT   = 28
LR=1e-4

class VOC2012ImagesOnly(VOCDetection):
    def __init__(self, root, year='2012', image_set='train', transform=None):
        super().__init__(root=root,
                         year=year,
                         image_set=image_set,
                         download=False,
                         transform=transform,
                         target_transform=None)
    def  __getitem__(self, idx):
        img, _ = super().__getitem__(idx)  # img 是 PIL.Image
    # 此时 transform 应该已经在父类 __getitem__ 中应用了，不需要再变换
        return img
    
class CIFAR100ImagesOnly(CIFAR100):
    def __init__(self, root, transform = None, target_transform = None):
        super().__init__(root=root,
                         train=True,
                         download=True,
                         transform=transform,
                         target_transform=None )
    def __getitem__(self, index):
        img,_=super().__getitem__(index)

        return img


# 2. 图像预处理：Resize -> CenterCrop -> ToTensor -> Normalize
transform = transforms.Compose([
    transforms.Resize((256, 256)),        # 缩放短边到256
    transforms.CenterCrop(224),           # 中心裁剪到224×224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 3. 实例化 Dataset & DataLoader

# dataset & dataloader
content_dataset = VOC2012ImagesOnly(root='data', transform=transform)
content_loader = DataLoader(content_dataset, batch_size=1, shuffle=True, num_workers=0,drop_last=True)
#在演示时batchsize改为1，训练时是16.

class StyleDataset(Dataset):
    def __init__(self, root, transform=None):
        # 获取所有图像文件
        self.files = list(sorted(Path(root).glob('*.jpg')))  # 你可以根据需要修改文件格式，如 *.png
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.files)
style_dataset=StyleDataset(root='data/style_data/painters_by_numbers',transform=transform)
style_loader = DataLoader(
    style_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True
)


