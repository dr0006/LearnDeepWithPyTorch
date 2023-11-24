# -*- coding: utf-8 -*-
"""
@File  : mnist_picdown.py
@author: FxDr
@Time  : 2023/11/25 0:52
@Description:把Mnist的测试集1w张转为图片形式
"""
# 导入包
import struct
import numpy as np
from PIL import Image


class MnistParser:
    # 加载图像
    def load_image(self, file_path):

        # 读取二进制数据
        binary = open(file_path, 'rb').read()

        # 读取头文件
        fmt_head = '>iiii'
        offset = 0

        # 读取头文件
        magic_number, images_number, rows_number, columns_number = struct.unpack_from(fmt_head, binary, offset)

        # 打印头文件信息
        print('图片数量:%d,图片行数:%d,图片列数:%d' % (images_number, rows_number, columns_number))

        # 处理数据
        image_size = rows_number * columns_number
        fmt_data = '>' + str(image_size) + 'B'
        offset = offset + struct.calcsize(fmt_head)

        # 读取数据
        images = np.empty((images_number, rows_number, columns_number))
        for i in range(images_number):
            images[i] = np.array(struct.unpack_from(fmt_data, binary, offset)).reshape((rows_number, columns_number))
            offset = offset + struct.calcsize(fmt_data)
            # 每1万张打印一次信息
            if (i + 1) % 10000 == 0:
                print('> 已读取:%d张图片' % (i + 1))

        # 返回数据
        return images_number, rows_number, columns_number, images

    # 加载标签
    def load_labels(self, file_path):
        # 读取数据
        binary = open(file_path, 'rb').read()

        # 读取头文件
        fmt_head = '>ii'
        offset = 0

        # 读取头文件
        magic_number, items_number = struct.unpack_from(fmt_head, binary, offset)

        # 打印头文件信息
        print('标签数:%d' % items_number)

        # 处理数据
        fmt_data = '>B'
        offset = offset + struct.calcsize(fmt_head)

        # 读取数据
        labels = np.empty(items_number)
        for i in range(items_number):
            labels[i] = struct.unpack_from(fmt_data, binary, offset)[0]
            offset = offset + struct.calcsize(fmt_data)
            # 每1万张打印一次信息
            if (i + 1) % 10000 == 0:
                print('> 已读取:%d个标签' % (i + 1))

        # 返回数据
        return items_number, labels

    # 图片可视化
    def visualaztion(self, images, labels, path):
        d = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        for i in range(images.__len__()):
            im = Image.fromarray(np.uint8(images[i]))
            im.save(path + "%d_%d.png" % (labels[i], d[labels[i]]))
            d[labels[i]] += 1
            # im.show()

            if (i + 1) % 10000 == 0:
                print('> 已保存:%d个图片' % (i + 1))


# 保存为图片格式
def change_and_save():
    mnist = MnistParser()

    # trainImageFile = './train-images.idx3-ubyte'
    # _, _, _, images = mnist.load_image(trainImageFile)
    # trainLabelFile = './train-labels.idx1-ubyte'
    # _, labels = mnist.load_labels(trainLabelFile)
    # mnist.visualaztion(images, labels, "./images/train/")

    testImageFile = r'X:\Coding\Github\data\MNIST\raw\t10k-images-idx3-ubyte'
    _, _, _, images = mnist.load_image(testImageFile)
    testLabelFile = r'X:\Coding\Github\data\MNIST\raw\t10k-labels-idx1-ubyte'
    _, labels = mnist.load_labels(testLabelFile)
    mnist.visualaztion(images, labels, "./data/images/test/")


# 测试
if __name__ == '__main__':
    change_and_save()
