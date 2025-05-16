
import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder network for style transfer. Takes AdaIN-processed features of shape [B, 512, 28, 28]
    and reconstructs an RGB image of shape [B, 3, 224, 224].
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            # input: [B, 512, 28, 28]
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 28 -> 56

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 56 -> 112

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 112 -> 224

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            # output: [B, 3, 224, 224]
        )

    def forward(self, x):
        return self.model(x)



# 示例：
# decoder = decoder()\# B, C, H, W = 1, 512, 28, 28
# input_feat = torch.randn(B, C, H, W)
# out_img = decoder(input_feat)  # shape [B, 3, 224, 224]

