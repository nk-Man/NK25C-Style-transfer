# NK25C-Style-transfer
作为南开大学25年C++课程的大作业设计。实现基于VGG网络的图像风格迁移任务。
基于pytorch框架在python端实现模型设计和保存
使用libtorch，openCV实现模型在C++端的部署
基于QT框架开发用户GUI
技术原理：使用了Adaptive Layer Normalization（自适应参数调整方法），对内容图和风格图的深层特征进行融合，解码还原出RGB图像并展示

查看源代码请直接看main分支。
根目录下存放的.py文件根据文件名，分别是训练脚本，工具文件（存放数据集加载器），单张图片验证脚本，展示脚本，导出脚本，超参数存放文件。
Model文件夹下存放模型的源代码
CPP文件夹下存放所有在qt平台上编写的源代码，即模型在C++端的部署加载和用户GUI的设计

请互评的同学不要误伤，这是C++项目。
如有疑问，请查看大作业报告文档或者查看b站视频，也可以飞书咨询 2410550 杨博
感谢查看
