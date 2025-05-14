# NK25C-Style-transfer
作为南开大学25年C++课程的大作业设计。实现基于VGG网络的图像风格迁移任务。
基于pytorch框架在python端实现模型设计和保存
使用libtorch，openCV实现模型在C++端的部署
基于QT框架开发用户GUI
技术原理：使用了Adaptive Layer Normalization（自适应参数调整方法），对内容图和风格图的深层特征进行融合，解码还原出RGB图像并展示

目前查看源代码请看分支 repo 3.0，因未上传文件夹，所以文件比较乱。后续会上传整个工作目录到main分支，
