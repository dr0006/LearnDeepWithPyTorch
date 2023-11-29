# -*- coding: utf-8 -*-
"""
@File  : config.py
@author: FxDr
@Time  : 2023/11/23 23:16
@Description:
"""
import torch
import torchvision.transforms as transforms  # transforms 模块包含用于图像预处理的各种转换操作

# 训练参数
num_epochs = 10  # 轮次
opt_func = torch.optim.Adam  # 优化器
lr = 5.5e-5  # 学习率
batch_size = 64  # 批次大小
# split_sizes = [55000, 500, 10000]  # 训练集验证集测试集图片数

# 转换器，将数据转换为tensor并归一化
# 数据增强防止过拟合
transformers = {
    # 原始处理
    'original': transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    'demo1': transforms.Compose([
        transforms.Resize((28, 28)),  # 其实不需要这一步，因为数据集已经是被处理好了的，都是28x28
        # 随机图像亮度、对比度、饱和度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # 随机翻转
        # transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        # 随机放射变化
        transforms.RandomAffine(degrees=11, translate=(0.1, 0.1), scale=(0.8, 0.8)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),

}
