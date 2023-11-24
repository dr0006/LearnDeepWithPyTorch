# -*- coding: utf-8 -*-
"""
@File  : UI_main.py
@author: FxDr
@Time  : 2023/11/24 14:34
@Description:
"""
import os
import sys
from datetime import datetime

from PyQt5.QtGui import QTextOption
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QTextEdit

from code.hand_write_CNN.model import ResNet, to_device, get_default_device
from code.hand_write_CNN.tools.predict_img import predict_img


class PredictionApp(QWidget):
    def __init__(self, model):
        super().__init__()
        self.btn_batch_predict = None
        self.btn_save_records = None
        self.text_edit_records = None
        self.lbl_result = None
        self.btn_select = None
        self.model = model
        self.initUI()

    def initUI(self):
        self.setWindowTitle('数字预测应用')
        self.setGeometry(100, 100, 600, 400)

        # 选择文件按钮
        self.btn_select = QPushButton('选择图片', self)
        self.btn_select.clicked.connect(self.selectImage)

        # 预测结果展示
        self.lbl_result = QLabel('预测结果：', self)

        # 识别记录展示（使用 QTextEdit 替代 QLabel）
        self.text_edit_records = QTextEdit(self)
        self.text_edit_records.setReadOnly(True)
        self.text_edit_records.setWordWrapMode(QTextOption.NoWrap)  # 禁用自动换行

        # 保存记录按钮
        self.btn_save_records = QPushButton('保存记录', self)
        self.btn_save_records.clicked.connect(self.saveRecords)

        # 批量识别按钮
        self.btn_batch_predict = QPushButton('批量识别', self)
        self.btn_batch_predict.clicked.connect(self.batchPredict)

        # 设置布局
        layout = QVBoxLayout(self)
        layout.addWidget(self.btn_select)
        layout.addWidget(self.btn_batch_predict)
        layout.addWidget(self.lbl_result)
        layout.addWidget(self.text_edit_records)  # 使用 QTextEdit
        layout.addWidget(self.btn_save_records)

    def selectImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数字图片", "",
                                                   "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)

        if file_path:
            # 进行预测
            result, records = predict_img(file_path, self.model)

            # 显示预测结果
            self.lbl_result.setText(f'预测结果：{result}')

            # 更新识别记录
            current_records = self.text_edit_records.toPlainText()
            current_records += f'\n{file_path} - {result} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            self.text_edit_records.setPlainText(current_records)

    def saveRecords(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(self, "保存识别记录", "", "Text Files (*.txt);;All Files (*)",
                                                   options=options)

        if file_path:
            # 将识别记录保存到文件
            with open(file_path, 'w') as f:
                f.write(self.text_edit_records.toPlainText())

    def batchPredict(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹", options=options)

        if folder_path:
            # 获取文件夹中的所有图片文件
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            # 遍历图片文件进行批量预测
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                result, records = predict_img(image_path, self.model)

                # 显示预测结果
                self.lbl_result.setText(f'预测结果：{result}')

                # 更新识别记录
                current_records = self.text_edit_records.toPlainText()
                current_records += f'\n{image_path} - {result} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                self.text_edit_records.setPlainText(current_records)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    model = ResNet()
    model.load_model_dict('./model/98.71%_model_weights.pth')
    model = to_device(model, get_default_device())
    window = PredictionApp(model)
    window.show()
    sys.exit(app.exec_())
