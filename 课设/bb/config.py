# -*- coding: utf-8 -*-
"""
@File  : config.py
@author: FxDr
@Time  : 2023/12/27 17:09
@Description:
"""
import torch
import torchvision

batch_size = 128
num_epochs = 20
# lr = 5.5e-5
lr = 6e-5
opt_func = torch.optim.Adam

# 图像增广
IMAGE_SIZE = 32

mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]

transform_train = torchvision.transforms.Compose([
    # 在高度和宽度上将图像RESIZE
    torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize
    torchvision.transforms.RandomHorizontalFlip(0.1),
    torchvision.transforms.RandomRotation(20),
    # torchvision.transforms.ColorJitter(brightness=0.1,  # 随机颜色抖动
    #                                    contrast=0.1,
    #                                    saturation=0.1),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize(mean, std)])

# 在测试期间，只对图像执行标准化，以消除评估结果中的随机性。
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)])

from torchvision import datasets
import torch
from torch.utils.data import DataLoader

torch.manual_seed(1)

# load and transform
train_dataset = datasets.CIFAR10(root='../../data', train=True, download=False, transform=transform_train)
validation_dataset = datasets.CIFAR10(root='../../data', train=False, download=False, transform=transform_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size)


def print_data_info():
    # 训练数据集信息
    print("训练数据集:")
    print("批次数量:", len(train_loader))
    print("总样本数量:", len(train_loader.dataset))
    print("一个批次的数据形状:", next(iter(train_loader))[0].shape)
    print("一个批次的标签形状:", next(iter(train_loader))[1].shape)

    # 验证数据集信息
    print("\n验证数据集:")
    print("批次数量:", len(validation_loader))
    print("总样本数量:", len(validation_loader.dataset))
    print("一个批次的数据形状:", next(iter(validation_loader))[0].shape)
    print("一个批次的标签形状:", next(iter(validation_loader))[1].shape)

    print("\n")
    print("--------------------------------")
    print(train_dataset.class_to_idx)
    print(len(train_dataset.classes))
