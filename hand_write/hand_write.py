import numpy as np
import paddle  # pip install paddlepaddle
from paddle.vision.transforms import Normalize
from PIL import Image
import os

# 设置为动态图模式
paddle.disable_static()

# 定义数据读取时的转换操作
# 对图像进行归一化，即减去均值然后除以标准差
transform = Normalize(mean=[127.5], std=[127.5])
# 加载MNIST训练集数据
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
# 加载MNIST测试集数据
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)


# 定义多层感知器模型
def multilayer_perceptron():
    return paddle.nn.Sequential(
        paddle.nn.Flatten(),  # 将图片矩阵展平
        # 784 = 28 * 28
        paddle.nn.Linear(784, 100),  # 第一个全连接层
        paddle.nn.ReLU(),  # ReLU激活函数
        paddle.nn.Linear(100, 100),  # 第二个全连接层
        paddle.nn.ReLU(),  # ReLU激活函数
        paddle.nn.Linear(100, 10)  # 第三个全连接层，输出层
    )


# 创建模型实例
model = multilayer_perceptron()

# 设置训练相关参数
learning_rate = 0.001  # 学习率
EPOCH_NUM = 1  # 训练轮数
BATCH_SIZE = 128  # 每批数据大小
model_save_dir = "hand_inference_model"  # 模型保存路径

# 设置优化器
optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())

# 训练过程
for epoch in range(EPOCH_NUM):
    model.train()  # 设置模型为训练模式
    # 使用DataLoader进行批处理读取数据
    for batch_id, data in enumerate(paddle.io.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)):
        x_data = data[0]  # 输入数据
        y_data = paddle.to_tensor(data[1].numpy().astype('int64').reshape(-1, 1))  # 标签数据

        predicts = model(x_data)  # 模型预测
        loss = paddle.nn.functional.cross_entropy(predicts, y_data)  # 计算损失
        acc = paddle.metric.accuracy(predicts, y_data)  # 计算准确率
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新参数
        optimizer.clear_grad()  # 清除梯度

        # 每100个批次输出一次训练状态
        if batch_id % 100 == 0:
            print(f"Epoch:{epoch}, Batch:{batch_id}, Loss:{float(loss)}, Accuracy:{float(acc)}")

    # 保存模型参数
    paddle.save(model.state_dict(), os.path.join(model_save_dir, 'mnist.pdparams'))

# 测试过程
model.eval()  # 设置模型为评估模式
accuracies = []  # 准确率列表
losses = []  # 损失列表
# 使用DataLoader进行批处理读取数据
for batch_id, data in enumerate(paddle.io.DataLoader(test_dataset, batch_size=BATCH_SIZE)):
    x_data = data[0]  # 输入数据
    y_data = paddle.to_tensor(data[1].numpy().astype('int64').reshape(-1, 1))  # 标签数据

    predicts = model(x_data)  # 模型预测
    loss = paddle.nn.functional.cross_entropy(predicts, y_data)  # 计算损失
    acc = paddle.metric.accuracy(predicts, y_data)  # 计算准确率
    accuracies.append(float(acc))  # 收集准确率
    losses.append(float(loss))  # 收集损失

# 计算测试集的平均准确率和损失
avg_acc = np.mean(accuracies)
avg_loss = np.mean(losses)
print(f"Test Loss:{avg_loss}, Test Accuracy:{avg_acc}")


# 推理函数
def load_image(img_path):
    img = Image.open(img_path).convert('L')  # 打开图像并转换为灰度图像
    img = img.resize((28, 28), Image.Resampling.LANCZOS)  # 重置图像大小
    img = np.array(img).astype('float32')  # 将图像转换为numpy数组
    img = img.reshape([1, 1, 28, 28])  # 重新塑形以匹配输入格式
    img = img / 255.0 * 2.0 - 1.0  # 归一化处理
    return img


# 预测示例函数
def infer(model, params_file_path, infer_path):
    param_dict = paddle.load(params_file_path)  # 加载模型参数
    model.load_dict(param_dict)  # 将参数载入模型
    model.eval()  # 设置模型为评估模式
    tensor_img = load_image(infer_path)  # 加载待推理的图像
    result = model(paddle.to_tensor(tensor_img))  # 进行预测
    return np.argmax(result.numpy())  # 返回预测结果的最大值索引，即预测的类别


# 加载模型参数并进行预测
infer_path = r'X:\Coding\Github\LearnDeepWithPyTorch\data\images\test\0_7.png'  # 待推理图像的路径
params_file_path = os.path.join(model_save_dir, 'mnist.pdparams')  # 模型参数路径
prediction = infer(model, params_file_path, infer_path)  # 进行预测
print(f"推测的图像标签为: {prediction}")  # 输出预测结果
