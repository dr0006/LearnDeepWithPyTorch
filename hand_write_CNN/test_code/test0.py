# -*- coding: utf-8 -*-
"""
@File  : test0.py
@author: FxDr
@Time  : 2023/11/25 0:41
@Description:
"""
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from hand_write_CNN.config import transformers
from hand_write_CNN.model import ResNet, get_default_device, to_device

test_dataset = MNIST(root='../../data', train=False, transform=transformers['original'], download=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 创建 ResNet 模型实例
device = get_default_device()
print("Using device:{}".format(device))
model = ResNet()
model.load_model_dict('../model/98.71%_model_weights.pth')
model = to_device(model, device)

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
