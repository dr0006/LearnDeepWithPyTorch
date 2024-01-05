# -*- coding: utf-8 -*-
"""
@File  : train2.py
@author: FxDr
@Time  : 2023/12/27 18:46
@Description:
"""
from 课设.bb.config import train_loader, validation_loader, num_epochs, lr, opt_func, print_data_info
from 课设.bb.model import MyResNet, get_default_device, DeviceDataLoader, to_device
from 课设.bb.tools import plot_accuracies, plot_losses

if __name__ == "__main__":
    # 创建 ResNet 模型实例
    model = MyResNet(3, 10)
    model.print_dataset()
    print("dataset")
    print_data_info()
    device = get_default_device()
    print("Using device:{}".format(device))
    train_dl = DeviceDataLoader(train_loader, device)
    val_dl = DeviceDataLoader(validation_loader, device)
    to_device(model, device)

    history = model.fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

    plot_accuracies(history)
    plot_losses(history)

    save_model = input("是否保存模型？ (yes/no): ").lower()
    if save_model == 'yes':
        model.save_best_model()
        print("Model saved")
    else:
        print("Model is not saved")
