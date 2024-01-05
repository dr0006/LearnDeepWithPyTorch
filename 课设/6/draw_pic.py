# -*- coding: utf-8 -*-
"""
@File  : draw_pic.py
@author: FxDr
@Time  : 2023/12/27 23:30
@Description:
"""
import pandas as pd
import matplotlib.pyplot as plt

# 读取导出的数据
train_accuracy_data = pd.read_csv('./csv_plot/AccuracyOfTrain.csv')
validation_accuracy_data = pd.read_csv('./csv_plot/AccuracyOfValidation.csv')
train_loss_data = pd.read_csv('./csv_plot/LossOfTrain.csv')
validation_loss_data = pd.read_csv('./csv_plot/LossOfValidation.csv')

# 检查列名
print("Train Accuracy Columns:", train_accuracy_data.columns)
print("Validation Accuracy Columns:", validation_accuracy_data.columns)
print("Train Loss Columns:", train_loss_data.columns)
print("Validation Loss Columns:", validation_loss_data.columns)

# 根据实际列名绘制图形
plt.figure(figsize=(12, 12))

# 绘制准确率图形
plt.subplot(2, 1, 1)
plt.plot(train_accuracy_data['Step'], train_accuracy_data['Value'], label='Training Accuracy', linestyle='-',
         marker='o')
plt.plot(validation_accuracy_data['Step'], validation_accuracy_data['Value'], label='Validation Accuracy',
         linestyle='-', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 绘制损失图形
plt.subplot(2, 1, 2)
plt.plot(train_loss_data['Step'], train_loss_data['Value'], label='Training Loss', linestyle='-', marker='o')
plt.plot(validation_loss_data['Step'], validation_loss_data['Value'], label='Validation Loss', linestyle='-',
         marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()  # 自动调整子图参数，确保图形不重叠
# plt.savefig('combined_plot.png')
plt.show()
