# -*- coding: utf-8 -*-
"""
@File  : resnet.py
@author: FxDr
@Time  : 2023/12/27 20:50
@Description: ResNet模型的定义，包括BasicBlock和Bottleneck两种基本块，以及ResNet主体结构的定义。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """
        BasicBlock基本块的定义
        Args:
            in_planes (int): 输入通道数
            planes (int): 输出通道数
            stride (int): 步长，默认为1
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        """
        前向传播
        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 输出张量
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        """
        Bottleneck基本块的定义
        Args:
            in_planes (int): 输入通道数
            planes (int): 输出通道数
            stride (int): 步长，默认为1
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        """
        前向传播
        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 输出张量
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        """
        ResNet主体结构的定义
        Args:
            block (nn.Module): 使用的基本块，可以是BasicBlock或Bottleneck
            num_blocks (list): 每个阶段使用的基本块数量的列表
            num_classes (int): 分类数，默认为10
        """
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        创建一个阶段的层
        Args:
            block (nn.Module): 使用的基本块
            planes (int): 输出通道数
            num_blocks (int): 基本块数量
            stride (int): 步长

        Returns:
            nn.Sequential: 一个阶段的层
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播
        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 输出张量
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    """
    创建ResNet18模型
    Returns:
        ResNet: ResNet18模型
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    """
    创建ResNet34模型
    Returns:
        ResNet: ResNet34模型
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    """
    创建ResNet50模型
    Returns:
        ResNet: ResNet50模型
    """
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    """
    创建ResNet101模型
    Returns:
        ResNet: ResNet101模型
    """
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    """
    创建ResNet152模型
    Returns:
        ResNet: ResNet152模型
    """
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    """
    用于测试的函数
    """
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
