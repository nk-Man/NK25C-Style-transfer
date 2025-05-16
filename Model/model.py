from src.AdaIN import adain#单独运行这个文件时，如果报错找不到路径，就把src.去掉
from src.Encoder import get_encoder
from src.Decoder import Decoder
import torch
import torch.nn as nn

class StyleTransferNet(nn.Module):
    def __init__(self, decoder_ckpt="decoder333.pth", device="cuda"):
        super().__init__()
        self.device = device

        # 1) Encoder：取前 30 层，用 eval() 冻结
        vgg = get_encoder()
        self.encoder = vgg.features[:30].eval().to(device)
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 2) Decoder：实例化并加载权重
        self.decoder = Decoder().to(device)
        
        # 1) 先加载原始 state dict
        orig_state = torch.load(decoder_ckpt, map_location=device)

        # 3) 用新的 state_dict 加载
        self.decoder.load_state_dict(orig_state)
        # 3) AdaIN 操作直接使用函数，不需 load
        self.adain=adain
      
    def forward(self, content, style):
        # content, style: [B,3,H,W]，预处理后!也就是说输入模型的必须是经过resize和归一化的tensor
        content = content.to(self.device)
        style   = style.to(self.device)

        # 1) 取特征
        f_c = self.encoder(content)
        f_s = self.encoder(style)

        # 2) AdaIN 融合
        t = self.adain(f_c, f_s)

        # 3) Decoder
        output = self.decoder(t)
        return output
    
if __name__=='__main__':
    
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=StyleTransferNet(decoder_ckpt='../decoder222.pth',device=device)#这里记得在交互端修改工作目录
    print("模型加载成功")
    print(model)
    print(model.parameters())
    model.eval()

