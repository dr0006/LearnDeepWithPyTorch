# -*- coding: utf-8 -*-
"""
@File  : train.py
@author: FxDr
@Time  : 2023/12/27 20:50
@Description: Training script with visualization and evaluation metrics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch.nn import DataParallel
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from resnet import ResNet18
# from ResNet50 import MyResNet
from CNN_V1 import CNN_V1

if __name__ == "__main__":

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0

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
        root='../../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4)

    validation_dataset = torchvision.datasets.CIFAR10(
        root='../../data', train=False, download=True, transform=transform_test)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=100, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # ---------------------  Model Net -------------------------------
    # 构建模型
    print('==> Building model..')
    # net = ResNet18()
    # net = MyResNet(3, 10)
    net = CNN_V1(out_1=32, out_2=64, out_3=128, number_of_classes=10, p=0.5)
    # ----------------------------------------------------------------

    # 使用DataParallel在多GPU上运行
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = DataParallel(net)

    net = net.to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # 定义学习率调整策略
    scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    # TensorBoard可视化
    # writer = SummaryWriter()
    writer = SummaryWriter(comment="CNN_V1-23.12.28")

    # 训练循环
    for epoch in range(start_epoch, start_epoch + 200):
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_accuracy = 100. * correct / total

        # 验证模型
        net.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(validation_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_accuracy = 100. * correct / total

        print("--> Epoch Number : {}".format(epoch + 1),
              " | Training Loss : {}".format(round(train_loss / (batch_idx + 1), 4)),
              " | Validation Loss : {}".format(round(val_loss / (batch_idx + 1), 4)),
              " | Training Accuracy : {}%".format(round(train_accuracy, 2)),
              " | Validation Accuracy : {}%".format(round(val_accuracy, 2)))

        # 写入TensorBoard
        writer.add_scalar('Loss/train', train_loss / (batch_idx + 1), epoch + 1)
        writer.add_scalar('Loss/validation', val_loss / (batch_idx + 1), epoch + 1)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch + 1)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch + 1)

        # 保存模型
        if val_accuracy > best_acc:
            print('Saving model..')
            state = {
                'net': net.state_dict(),
                'acc': val_accuracy,
                'epoch': epoch,
            }
            torch.save(state, 'best_model.pth')
            best_acc = val_accuracy

        # 调整学习率
        scheduler.step()

    writer.close()  # 关闭TensorBoard写入

    # 加载最好的模型并进行测试
    print("Loading the best model for testing..")
    # best_model = ResNet18()
    # best_model = MyResNet(3, 10)
    best_model = CNN_V1(out_1=32, out_2=64, out_3=128, number_of_classes=10, p=0.5)

    best_model = best_model.to(device)
    best_model.load_state_dict(torch.load('best_model.pth')['net'])
    best_model.eval()

    test_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

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

    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)

    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    print("Training finished.")
