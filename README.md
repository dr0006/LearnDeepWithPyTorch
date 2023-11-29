# LearnDeepWithPyTorch

学习深度学习，主要是PyTorch

----

学习通作业  
[workForStu](./code/workForStu)

1. 手写数字识别(前馈神经网络)
2. MINST(深度学习网络)


### MNIST数据集

    共有训练数据60000项、测试数据10000项。
    每张图像的大小为28*28（像素），每张图像都为灰度图像，位深度为8（灰度图像是0-255）。  

由Yann LeCun搜集，是一个大型的手写体数字数据库，通常用于训练各种图像处理系统，也被广泛用于机器学习领域的训练和测试。  

'''
1. train-images.idx3-ubyte.gz：训练集图片（9912422字节），55000张训练集，5000张验证集
2. train-labels.idx1-ubyte.gz：训练集图片对应的标签（28881字节），
3. t10k-images.idx3-ubyte .gz：测试集图片（1648877字节），10000张图片
4. t10k-labels.idx1-ubyte.gz：测试集图片对应的标签（4542字节）
'''

