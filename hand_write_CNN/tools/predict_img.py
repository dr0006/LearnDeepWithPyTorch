# -*- coding: utf-8 -*-
"""
@File  : predict_img.py
@author: FxDr
@Time  : 2023/11/24 10:54
@Description:
"""
import torch
from PIL import Image

from hand_write_CNN.config import transformers
from hand_write_CNN.model import get_default_device, to_device
from hand_write_CNN.tools.my_tools import class_index

import matplotlib.pyplot as plt

# 中文绘制
plt.rcParams['font.family'] = 'Microsoft YaHei'


def predict_img(path, model):
    """
        :param 将传入路径读取为灰度图像，然后移到GPU cuda0 上，再用modeL进行预测
        :return 预测标签和准确度
    """
    device = get_default_device()
    image = Image.open(path).convert('L')  # 转为单通道灰度图像！！！！
    # 设定阈值，将像素值小于阈值的设置为0（黑色），大于等于阈值的设置为255（白色）
    threshold = 128
    image = image.point(lambda p: p > threshold and 255)
    image_clean = transformers['original'](image)
    # 将图像转换为张量并移动到device
    xb = to_device(image_clean.unsqueeze(0), device)
    # 预测
    yb = model(xb)
    # 选择概率最高的类别
    prob, preds = torch.max(yb, dim=1)

    label = class_index[preds[0].item()]  # 预测标签 数字0-9
    plt.title(label)
    plt.imshow(image_clean.squeeze().numpy(), cmap='gray')
    plt.show()
    return label, prob.item()
