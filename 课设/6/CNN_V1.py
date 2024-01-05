# -*- coding: utf-8 -*-
"""
@File  : CNN_V1.py
@author: FxDr
@Time  : 2023/12/28 0:37
@Description:
"""
import torch.nn as nn
import torch.nn.functional as F


class CNN_V1(nn.Module):
    """
    添加一个隐藏层、调整 dropout 值、增加一个卷积层
    总共 3 个隐藏层、3 个卷积层和批量归一化
    """

    # 构造函数
    def __init__(self, out_1=32, out_2=64, out_3=128, number_of_classes=10, p=0):
        super(CNN_V1, self).__init__()

        # 第一个卷积层
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=out_1, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(out_1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.drop_conv = nn.Dropout(p=0.2)

        # 第二个卷积层
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, padding=2)
        self.conv2_bn = nn.BatchNorm2d(out_2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # 第三个卷积层
        self.cnn3 = nn.Conv2d(in_channels=out_2, out_channels=out_3, kernel_size=5, padding=2)
        self.conv3_bn = nn.BatchNorm2d(out_3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # 隐藏层 1
        self.fc1 = nn.Linear(out_3 * 4 * 4, 1000)
        self.fc1_bn = nn.BatchNorm1d(1000)
        self.drop = nn.Dropout(p=p)

        # 隐藏层 2
        self.fc2 = nn.Linear(1000, 1000)
        self.fc2_bn = nn.BatchNorm1d(1000)

        # 隐藏层 3
        self.fc3 = nn.Linear(1000, 1000)
        self.fc3_bn = nn.BatchNorm1d(1000)

        # 隐藏层 4
        self.fc4 = nn.Linear(1000, 1000)
        self.fc4_bn = nn.BatchNorm1d(1000)

        # 最终输出层
        self.fc5 = nn.Linear(1000, number_of_classes)
        self.fc5_bn = nn.BatchNorm1d(number_of_classes)

    # 前向传播
    def forward(self, x):
        x = self.cnn1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.drop_conv(x)

        x = self.cnn2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.drop_conv(x)

        x = self.cnn3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        x = self.drop_conv(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(self.drop(x))

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(self.drop(x))

        x = self.fc3(x)
        x = self.fc3_bn(x)
        x = F.relu(self.drop(x))

        x = self.fc4(x)
        x = self.fc4_bn(x)
        x = F.relu(self.drop(x))

        x = self.fc5(x)
        x = self.fc5_bn(x)

        return x

# model = CNN_V1(out_1=32, out_2=64, out_3=128, number_of_classes=10, p=0.5)
