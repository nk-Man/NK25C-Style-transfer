# train2.py

import torch
from tqdm import tqdm
from src.model import StyleTransferNet
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True#加速进程
    
    # --------- 导入模块（放在 main 里避免 spawn 时重复执行顶层代码） ----------
    from utils import style_loader, content_loader, CONTENT_WEIGHT, STYLE_WEIGHT
    import torch.nn.functional as F

    # --------- 设备配置 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------- 模型准备 ----------
    # Encoder：只取前 30 层（到 relu4_1），冻结并 eval
    model  = StyleTransferNet(decoder_ckpt='./decoder.pth', device=device)
    encoder = model.encoder
    for p in encoder.parameters():
        p.requires_grad = False

    # Decoder：实例化、train 模式、搬到 device
    decoder = model.decoder
    decoder.train()

    adain=model.adain


    # --------- 优化器 & Loss 函数 ----------
    def get_optimizer(decoder, lr=1e-4, betas=(0.9, 0.999), weight_decay=0):
        """
        构建 Adam 优化器，只优化 decoder 的参数
        """
        optim=torch.optim.Adam(params=decoder.parameters(),lr=lr,betas=betas,weight_decay=weight_decay)
        return optim
    optimizer = get_optimizer(decoder, lr=1e-4)

    def calc_content_loss(gen_feat, content_feat):
        return F.mse_loss(gen_feat, content_feat)

    def calc_style_loss(gen_feats, style_feats):
        def mean_std(feat, eps=1e-5):
            B, C, H, W = feat.size()
            var = feat.view(B, C, -1).var(dim=2, unbiased=False) + eps
            std = var.sqrt().view(B, C, 1, 1)
            mean = feat.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
            return mean, std

        loss = 0.0
        for gf, sf in zip(gen_feats, style_feats):
            gm, gs = mean_std(gf)
            sm, ss = mean_std(sf)
            loss += F.mse_loss(gm, sm) + F.mse_loss(gs, ss)
        return loss

    # --------- 训练循环 ----------
    num_epochs = 30
    try:
        for epoch in range(num_epochs):
            running_loss = 0.0
            pbar = tqdm(total=len(content_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

            for content_batch in content_loader:
                style_iter=iter(style_loader)
                style_batch = next(style_iter,None)

                content_batch = content_batch.to(device, non_blocking=True,dtype=torch.float32)
                style_batch   = style_batch.to(device, non_blocking=True,dtype=torch.float32)

                # 编码
                content_feats = encoder(content_batch)
                style_feats   = encoder(style_batch)

                # AdaIN
                t = adain(content_feats, style_feats)

                # 解码 & 重编码
                gen_batch = decoder(t)
                gen_feats  = encoder(gen_batch)

                # 计算损失
                loss_c = calc_content_loss(gen_feats, content_feats)
                # 只在 relu4_1（当前唯一一层）计算风格损失，需用列表包装
                loss_s = calc_style_loss([gen_feats], [style_feats])

                loss   = CONTENT_WEIGHT * loss_c + STYLE_WEIGHT * loss_s

                # 优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (pbar.n + 1))
                pbar.update(1)

            pbar.close()
            avg = running_loss / len(content_loader)
            print(f"Epoch {epoch+1}/{num_epochs}  average loss: {avg:.4f}")

   
    except KeyboardInterrupt:
        print("检测到 Ctrl+C，中断训练并保存模型...")
        torch.save(decoder.state_dict(), "decoder222.pth")
        print("模型已保存为 decoder222.pth")
