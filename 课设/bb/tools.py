# -*- coding: utf-8 -*-
"""
@File  : tools.py
@author: FxDr
@Time  : 2023/11/12 21:28
@Description:
"""
import matplotlib.pyplot as plt

# 中文绘制
plt.rcParams['font.family'] = 'Microsoft YaHei'


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
