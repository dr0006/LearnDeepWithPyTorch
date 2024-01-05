# -*- coding: utf-8 -*-
"""
@File  : test2.py
@author: FxDr
@Time  : 2023/12/27 23:51
@Description:
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

from resnet import ResNet18
from ResNet50 import MyResNet
from CNN_V1 import CNN_V1

# 类别映射字典
class_mapping = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


def test_local_image(model, image_path):
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 读取图像并进行预处理
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = output.max(1)

    # 获取预测标签
    predicted_label = class_mapping[predicted.item()]

    # 显示预测的图片
    plt.imshow(image)
    plt.title(f'Predicted Class: {predicted_label} ({predicted.item()})')
    print("predicted_label", predicted_label)
    plt.axis('off')
    plt.show()


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

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 加载模型并进行测试
    print("Loading the best model for testing..")
    print("transform_train:", transform_train)
    print("transform_test:", transform_test)

    # """
    best_model = ResNet18()
    best_model = best_model.to(device)
    best_model.load_state_dict(torch.load('./model_file/95.08%_ResNet18_model.pth')['net'])
    print("Model Train of Net:", torch.load('./model_file/95.08%_ResNet18_model.pth')['acc'])
    print("The Model Epochs:")
    # """

    """
    best_model = MyResNet(3, 10)
    best_model = best_model.to(device)
    best_model.load_state_dict(torch.load('./model_file/85.90%_MyResNet_model.pth'))
    best_model.load_state_dict(torch.load('./model_file/93.29%_MyResNet_model.pth'))
    """

    """
    best_model = CNN_V1(out_1=32, out_2=64, out_3=128, number_of_classes=10, p=0.5)
    best_model = best_model.to(device)
    # best_model.load_state_dict(torch.load('./model_file/85.7%_CNNV1_model.pth'))
    # best_model.load_state_dict(torch.load('./model_file/88.75%_CNNV1_model.pth'))
    best_model.load_state_dict(torch.load('./model_file/88.98%_CNNV1_model.pth'))
    """

    # ------------------------------
    print("Model Name:")
    print(best_model.__class__)

    best_model.eval()

    # 进行测试
    # test_local_image(best_model, r'C:\Users\lenovo\Downloads\下载.jpg')
    test_local_image(best_model, r'X:\Coding\Github\LearnDeepWithPyTorch\课设\eg_IMG\img.png')
