from src.model import StyleTransferNet
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch
from utils import content_loader,content_dataset

if __name__=='__main__':
    # 1) 读图 & 预处理
    content_path = 'man.jpg'
    style_path='shout.jpg'
    content_img  = Image.open(content_path).convert("RGB")
    style_img=Image.open(style_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    Totensor=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    orig_tensor=Totensor(content_img).unsqueeze(0)
    # 3x224x224  -> add batch -> 1x3x224x224
    content_tensor = preprocess(content_img).unsqueeze(0)
    style_tensor=preprocess(style_img).unsqueeze(0)

    # 2) 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = StyleTransferNet(decoder_ckpt='./decoder444.pth', device=device)
    model.eval()

    # 3) 前向
    with torch.no_grad():
        output=model(content_tensor,style_tensor)

    # 4) 反归一化：把 output 从标准化空间还原到 [0,1]
    def unnormalization(output):
        output=output.to(device)
        mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
        x = output * std + mean
        x = torch.clamp(x, 0.0, 1.0)  # [1,3,224,224]
        return x
    
  
    # 5) 写入 TensorBoard：add_image 接受 3xHxW 的 tensor
    step=3
    writer = SummaryWriter(log_dir='logs')
    x=unnormalization(output)
    writer.add_image("原始内容图",    orig_tensor.cpu()[0], global_step=step)
    writer.add_image("风格迁移后",    x.cpu()[0],  global_step=step)
    writer.close()
    

#tensorboard --logdir=logs