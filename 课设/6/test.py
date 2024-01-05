# -*- coding: utf-8 -*-
"""
@File  : test.py
@author: FxDr
@Time  : 2023/12/27 23:43
@Description:
"""

from resnet import ResNet18

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 数据预处理
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='../../data', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=4)

    validation_dataset = torchvision.datasets.CIFAR10(
        root='../../data', train=False, download=False, transform=transform_test)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=256, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 加载最好的模型并进行测试
    print("Loading the best model for testing..")
    print("transform_train:", transform_train)
    print("transform_test:", transform_test)
    best_model = ResNet18()
    best_model = best_model.to(device)
    best_model.load_state_dict(torch.load('95.08%_ResNet18_model.pth')['net'])
    print("Model Train of Net:", torch.load('95.08%_ResNet18_model.pth')['acc'])
    best_model.eval()

    test_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validation_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = best_model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_accuracy = 100. * correct / total
    print('Test Accuracy: {:.2f}%'.format(test_accuracy))
