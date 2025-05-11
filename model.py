from Encoder import encoder
from AdalN import adaln
from Decoder import decoder


if __name__ =='main':
    content = preprocess(content_img)
    style = preprocess(style_img)

    "使用vgg网络提取内容图和风格图的特征"
    F_c = encoder(content)
    F_s = encoder(style)
    "使用adaln运算 生成输出图的特征"
    F_t = adaln(F_c, F_s)
    "使用上采样方法 对生成特征图解码出RGB图像"
    output = decoder(F_t)
