'Adaptive Instance Normalization'
import torch
import torch.nn.functional as F

def adaln(f_c:torch.Tensor,f_s:torch.Tensor,eps=1e-5)->torch.Tensor:
    """
        Adaptive Instance Normalization (AdaIN)

        Args:
            f_c: Tensor of shape (B, C, H, W) — features from the content image
            f_c:   Tensor of shape (B, C, H, W) — features from the style image
            eps: Small value to avoid division by zero

        Returns:
            stylized_feat: Tensor of shape (B, C, H, W)
        """
    f_c_mean=f_c.mean(dtype=torch.float64)
    f_c_std=f_c.std(dtype=torch.float64)+eps

    f_s_mean=f_s.mean(dtype=torch.float64)
    f_s_std=f_s.std(dtype=torch.float64)+eps

    normalized = (f_c - f_c_mean) / f_c_std
    stylized_feat = normalized * f_s_std + f_s_mean

    return stylized_feat