# -*- coding: utf-8 -*-
"""
@File  : train.py
@author: FxDr
@Time  : 2023/11/23 23:30
@Description:
"""
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST

from hand_write_CNN.config import transformers, batch_size, num_epochs, lr, opt_func
from hand_write_CNN.model import ResNet, get_default_device, DeviceDataLoader, to_device
from hand_write_CNN.tools.my_tools import plot_accuracies, plot_losses

if __name__ == '__main__':
    # 下载数据集
    train_dataset = MNIST(root='../data', train=True, transform=transformers['original'], download=False)
    test_dataset = MNIST(root='../data', train=False, transform=transformers['original'], download=False)  # 测试集

    # 划分训练集为训练集和验证集
    train_dataset, val_dataset = random_split(train_dataset, [55000, 5000])

    # 数据加载器
    # batch_size*2 验证集的批量大小是训练集的两倍
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=4)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 创建 ResNet 模型实例
    model = ResNet()

    # 移动到GPU
    device = get_default_device()
    print("Using device:{}".format(device))
    train_dl = DeviceDataLoader(train_loader, device)
    val_dl = DeviceDataLoader(val_loader, device)
    to_device(model, device)

    # training
    history = model.fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    # 可视化损失和准确率
    plot_accuracies(history)
    plot_losses(history)

    save_model = input("是否保存模型？ (yes/no): ").lower()
    if save_model == 'yes':
        model.save_model()
        print("Model saved")
    else:
        print("Model is not saved")
