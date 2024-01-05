# -*- coding: utf-8 -*-
"""
@File  : main.py
@author: FxDr
@Time  : 2024/01/05 22:47
@Description: resnet18 的 UI
"""
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import torch
import torchvision.transforms as transforms
from PIL import Image
from resnet import ResNet18

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

class_mapping_chinese = {
    'airplane': '飞机',
    'automobile': '汽车',
    'bird': '鸟',
    'cat': '猫',
    'deer': '鹿',
    'dog': '狗',
    'frog': '青蛙',
    'horse': '马',
    'ship': '船',
    'truck': '卡车'
}


class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()

        self.model = ResNet18()  # 默认模型
        self.predict_button = None
        self.choose_model_button = None
        self.choose_image_button = None
        self.result_label = None
        self.image_label = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('图像分类预测')
        self.setGeometry(100, 100, 600, 400)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(300, 300)

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFixedSize(300, 50)

        self.choose_image_button = QPushButton('选择本地图片', self)
        self.choose_image_button.clicked.connect(self.chooseImage)

        self.choose_model_button = QPushButton('选择模型文件', self)
        self.choose_model_button.clicked.connect(self.chooseModel)

        self.predict_button = QPushButton('进行预测', self)
        self.predict_button.clicked.connect(self.predict)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.choose_image_button)
        layout.addWidget(self.choose_model_button)
        layout.addWidget(self.predict_button)

        self.setLayout(layout)

    def chooseImage(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, '选择本地图片', '', 'Images (*.png *.jpg *.bmp *.jpeg)')
        if image_path:
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap)
            self.image_path = image_path

    def chooseModel(self):
        file_dialog = QFileDialog()
        model_path, _ = file_dialog.getOpenFileName(self, '选择模型文件', '', 'Model Files (*.pth)')
        if model_path:
            # 加载选择的模型文件
            self.model = ResNet18()
            self.model.load_state_dict(torch.load(model_path)['net'])
        # 如果没有选择文件，则不改变当前的模型实例

    def predict(self):
        if hasattr(self, 'image_path') and self.image_path:
            image = Image.open(self.image_path)
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            image_tensor = transform(image).unsqueeze(0)

            # 使用模型进行预测
            self.model.eval()
            with torch.no_grad():
                output = self.model(image_tensor)
                _, predicted = output.max(1)

            predicted_label_en = class_mapping[predicted.item()]
            predicted_label_ch = class_mapping_chinese[predicted_label_en]

            # 显示预测结果
            self.result_label.setText(f'预测标签(英文)：{predicted_label_en}\n预测标签(中文)：{predicted_label_ch}')
        else:
            self.result_label.setText('请先选择图片')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageClassifierApp()
    ex.show()
    sys.exit(app.exec_())
