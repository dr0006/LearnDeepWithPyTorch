# -*- coding: utf-8 -*-
"""
@File  : mean.py
@author: FxDr
@Time  : 2023/12/28 19:12
@Description:
"""
import warnings
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchvision import transforms, models

warnings.filterwarnings("ignore")

# 设置随机种子，以便实验的可重复性
np.random.seed(0)

# 加载预训练的ResNet18模型
model = models.resnet18(pretrained=True)
# 截取掉模型的最后一层（全连接层），获取特征
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

plt.rcParams['font.family'] = 'Microsoft YaHei'

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 选择训练集与测试集的数据，下载数据集
train_data = torchvision.datasets.CIFAR10('../../data', train=True,
                                          download=False, transform=transform_train)
test_data = torchvision.datasets.CIFAR10('../../data', train=False,
                                         download=False, transform=transform_test)

# 预测标签到类别名称的映射
label_mapping = {
    0: '类别1',
    1: '类别2',
    2: '类别3',
    3: '类别4',
    4: '类别5',
    5: '类别6',
    6: '类别7',
    7: '类别8',
    8: '类别9',
    9: '类别10'
}


# 提取特征的函数
def extract_features(data_loader, model, device):
    features = []
    labels = []
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            output = model(inputs)
        features.append(output.squeeze().cpu().numpy())
        labels.append(targets.cpu().numpy())
    return np.vstack(features), np.concatenate(labels)


# 使用提取特征的函数得到训练集和测试集的特征和标签
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

train_features, train_labels = extract_features(train_loader, model, device)
test_features, test_labels = extract_features(test_loader, model, device)

# 使用PCA进行降维
n_components = 200  # 设置PCA的维度
pca = PCA(n_components=n_components)
train_features_pca = pca.fit_transform(train_features)
test_features_pca = pca.transform(test_features)

# 使用KMeans进行聚类
n_clusters = 10  # 设置聚类的数量
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
train_clusters = kmeans.fit_predict(train_features_pca)
test_clusters = kmeans.predict(test_features_pca)

# 打印各聚类标签的各真实标签数量
cluster_label_counts = np.zeros((n_clusters, 10), dtype=int)
for cluster_label, true_label in zip(train_clusters, train_labels):
    cluster_label_counts[cluster_label, true_label] += 1

for i in range(n_clusters):
    print(f"Cluster {i}: {cluster_label_counts[i]}")
