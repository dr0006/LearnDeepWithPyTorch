# -*- coding: utf-8 -*-
"""
@File  : my_tools.py
@author: FxDr
@Time  : 2023/11/23 23:41
@Description:
"""
import matplotlib.pyplot as plt

# 中文绘制
plt.rcParams['font.family'] = 'Microsoft YaHei'

# 6
class_index = ['数字0', '数字1', '数字2', '数字3', '数字4', '数字5', '数字6', '数字7', '数字8', '数字9']


def plot_accuracies(history):
    """绘制准确率曲线"""
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()


def plot_losses(history):
    """绘制损失曲线"""
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()
