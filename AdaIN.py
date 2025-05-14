'Adaptive Instance Normalization'
import torch
import torch.nn.functional as F

def adain(f_c:torch.Tensor,f_s:torch.Tensor,eps=1e-5)->torch.Tensor:
    """
        Adaptive Instance Normalization (AdaIN)

        Args:
            f_c: Tensor of shape (B, C, H, W) — features from the content image
            f_c:   Tensor of shape (B, C, H, W) — features from the style image
            eps: Small value to avoid division by zero

        Returns:
            stylized_feat: Tensor of shape (B, C, H, W)
        """
    B, C, H, W = f_c.size()

    # 1) 计算每个通道的均值和标准差（在 H×W 空间维度）
    fc_mean = f_c.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
    fc_std  = f_c.view(B, C, -1).std(dim=2, unbiased=False).view(B, C, 1, 1) + eps

    fs_mean = f_s.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
    fs_std  = f_s.view(B, C, -1).std(dim=2, unbiased=False).view(B, C, 1, 1) + eps

    # 2) 归一化内容特征，再用风格的均值和标准差重新标度
    normalized = (f_c - fc_mean) / fc_std
    stylized  = normalized * fs_std + fs_mean

    return stylized