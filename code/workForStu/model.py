# -*- coding: utf-8 -*-
"""
@File  : model.py
@author: FxDr
@Time  : 2023/11/07 21:59
@Description:
"""
import torch.nn as nn


# hand_write
# 定义前馈神经网络模型
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self):
        super(FeedforwardNeuralNetModel, self).__init__()
        # 28x28 = 784
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()  # relu 激活函数
        self.fc2 = nn.Linear(128, 10)  # 输出10，0-9

    def forward(self, x):
        # Flatten the image
        x = x.view(-1, 784)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
