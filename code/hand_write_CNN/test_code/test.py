# -*- coding: utf-8 -*-
"""
@File  : test.py
@author: FxDr
@Time  : 2023/11/24 10:51
@Description:
"""
from code.hand_write_CNN.model import ResNet, get_default_device, to_device
from code.hand_write_CNN.tools.predict_img import predict_img

device = get_default_device()
print("Using device:{}".format(device))
model = ResNet()
model.load_model_dict('../model/98.71%_model_weights.pth')
model = to_device(model, device)

# path = r'X:\Coding\Github\LearnDeepWithPyTorch\code\MINST_test\4.png'
# path = r'X:\Coding\Github\LearnDeepWithPyTorch\code\MINST_test\infer_3.png'
path = r'/code/MNIST_test\1.png'

a, b = predict_img(path, model)
print("预测为:{}".format(a))
print(b)
