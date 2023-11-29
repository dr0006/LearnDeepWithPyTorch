# -*- coding: utf-8 -*-
"""
@File  : 2test1.py
@author: FxDr
@Time  : 2023/11/07 21:55
@Description:
"""

import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import torch.nn as nn


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


# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:{}".format(device))

# 转换器，将数据转换为tensor并归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载数据集假如没下载，root目录下存在就不用重复下载
test_dataset = MNIST(root='../data', train=False, transform=transform, download=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 加载模型
model = FeedforwardNeuralNetModel().to(device)
model.load_state_dict(torch.load('./model/hand_write_model.pth', map_location=device))
model.eval()

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += torch.Tensor(predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
