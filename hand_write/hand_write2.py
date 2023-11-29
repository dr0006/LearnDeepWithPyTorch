# -*- coding: utf-8 -*-
"""
@File  : hand_write2.py
@author: FxDr
@Time  : 2023/11/07 19:34
@Description:
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import matplotlib.pyplot as plt  # 可视化训练结果

# 设置超参数
batch_size = 64  # 批次大小
learning_rate = 0.01  # 学习率
num_epochs = 5  # 训练轮次

# 转换器，将数据转换为tensor并归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载数据集
train_dataset = MNIST(root='../data', train=True, transform=transform, download=False)
test_dataset = MNIST(root='../data', train=False, transform=transform, download=False)

# 划分训练集为训练集和验证集
train_dataset, val_dataset = random_split(train_dataset, [55000, 5000])

# 数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


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


# 实例化模型
# model = FeedforwardNeuralNetModel()
# 移动到gpu,假如有的话
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("使用{}".format(device))
model = FeedforwardNeuralNetModel().to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # learning_rate学习率

# 初始化损失和准确率的记录器
train_losses = []
val_accuracies = []

# 训练模型
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # 将数据和标签转移到设备上
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass 前向传播损失计算
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))

    # 评估阶段
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)
    print(f'Validation Accuracy: {val_accuracy:.2f}%')

# 可视化损失
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.legend()
plt.show()

# 可视化准确率
plt.figure(figsize=(10, 5))
plt.plot(val_accuracies, label='Validation accuracy')
plt.legend()
plt.show()

# 保存模型
torch.save(model.state_dict(), './hand_write_model.pth')
print('Saved PyTorch Model State to hand_write_model.pth')
