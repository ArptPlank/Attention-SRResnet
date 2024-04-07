---

# SRResnet改进版本

## 项目简介
对分辨率照片进行超分辨率放缩
本项目在原有的SRResnet基础上进行了改进，主要增加了以下功能：

- **通道注意力机制和空间注意力机制**：引入了注意力机制来进一步提升图像的超分辨率效果。
- **损失函数变更**：从均方差损失函数(MSE)更换为结构相似性损失函数(MiSSIM)，使得训练结果更加符合人类视觉感知。
- **参数量**：140余万

## 文件说明
- **predict.py**：该文件用于对输入的图片进行预测。运行后，请在对话框中输入文件地址，结果将在约0.5秒内返回（如果没有GPU，则需要更长时间）。
- **train_srresnet.py**：用于训练模型的脚本。以4070 Ti显卡为例，
  1400张520x320像素的图片每轮大约需要10分钟，总计算时间约1小时（因为有预训练模型），显存占用接近16GB。
- **get_image.py**（位于数据集爬取文件夹内）：可以修改搜索关键词来爬取不同类型的图片。建议在爬取后进行数据清洗以保证训练质量。
- **multi_scale_ssim.py**：定义了用于训练的结构相似性损失函数。
- **rresnet.py**：模型框架文件。

## 使用说明
- 运行训练脚本之前，请先将爬取到的数据集文件夹(dataset)直接移动到根目录的dataset文件夹内
- 确保环境配置正确，在放置好数据集后可直接运行`train_srresnet.py`进行模型训练。训练数据越多，效果可能越好，但计算时间会相应增加。
- 运行`predict.py`进行图像预测。注意，模型主要针对自然风光类图片进行优化，对于其他类型图片效果可能不佳。
- 每次开始新的训练会清空上一次的模型文件，如需保留，请提前备份。

## 效果展示
高分辨率原图
-
![高分辨率原图](test_image/origin.png)
-
低分辨率原图
-
![低分辨率原图](test_image/LR_image.png)
-
模型超分图像
-
![模型超分图像](test_image/output_image.png)
-
-
Loss值曲线
-
![loss](loss.svg)
-
---
