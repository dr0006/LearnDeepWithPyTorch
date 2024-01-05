# -*- coding: utf-8 -*-
"""
@File  : ResNet50.py
@author: FxDr
@Time  : 2023/12/28 0:10
@Description: 自定义简化版 ResNet50 模型
"""

import torch.nn as nn


def conv_block(in_channels, out_channels, pool=False):
    # 定义卷积块，包括卷积层、批归一化层和激活函数ReLU
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))  # 如果需要池化，则添加最大池化层
    return nn.Sequential(*layers)


class MyResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 定义模型的各个层
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        # 定义残差块1，包括两个卷积块
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        # 定义残差块2，包括两个卷积块
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        # 分类器部分，包括最大池化、展平、全连接层
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        # 前向传播过程
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out  # 添加残差连接
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out  # 添加残差连接
        out = self.classifier(out)
        return out
