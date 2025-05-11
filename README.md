# NK25C-Style-transfer
作为南开大学25年C++课程的大作业设计。实现基于VGG网络的图像风格迁移任务。
基于pytorch框架在python端实现模型设计和保存
使用libtorch，openCV实现模型在C++端的部署
基于QT框架开发用户GUI
技术原理：使用了Adaptive Layer Normalization（自适应参数调整方法），对内容图和风格图的深层特征进行融合，解码还原出RGB图像并展示


repo 1.0版本：
Encoder采用了vgg19模型特征提取器的所有层。有5个CONV block和5次maxpooling。若输入为[1,3,224,224]最终输出降为[1,512,2,2]。可见网络层次太深，导致图像特征提取过深，丢失了图像的空间特征而只提取了深层语义。效果不好
