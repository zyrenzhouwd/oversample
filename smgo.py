import pandas as pd
import numpy as np
import math
import random
from mrmr import mrmr_regression, mrmr_classif
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree, KNeighborsClassifier


class SMGO:
    def __init__(self, b=1, w=0.5, k1=5, k2=2, k3=5, r=1) -> None:
        self.w = w
        self.b = b
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.r = float(r)
        self.mode = 'random'

    def fit_resample(self, x: np.ndarray, y: np.ndarray):
        n_features = x.shape[1]
        # 将特征矩阵转换为Dataframe
        df = pd.DataFrame(x, columns=[i for i in range(n_features)])
        # indexes是mrmr得到的特征排序（元素是特征标号,从0开始）
        indexes = mrmr_classif(df, y, K=x.shape[1], show_progress=False)
        # print(indexes)
        #indexes = mrmr_regression(df, y, K=x.shape[1], show_progress=False)
        # 获取少量样本的标签
        label = 1 if y[y == 0].shape[0] >= y[y == 1].shape[0] else 0
        num_minor, num_major = y[y == label].shape[0], y[y != label].shape[0]
        #print('cc: {}，dd: {}'.format(num_minor,num_major))
        #print('minor: {}, major: {}'.format(num_minor, num_major))

        # 区分少量样本和大量样本
        x_minor, x_major = x[y == label], x[y != label]

        # 寻找少类边界
        maj_bl_ind, min_bl_ind = set(), set()
        maj_bl, min_bl = None, None
        kd_maj, kd_min = KDTree(x_major), KDTree(x_minor)
        for i in range(x_minor.shape[0]):
            _, ind = kd_maj.query(x_minor[i].reshape(1, -1), k=self.k1)
            maj_bl_ind |= set(ind.reshape(-1).tolist())
        maj_bl = x_major[list(maj_bl_ind)]
        for i in range(maj_bl.shape[0]):
            _, ind = kd_min.query(maj_bl[i].reshape(1, -1), k=self.k2)
            min_bl_ind |= set(ind.reshape(-1).tolist())
        min_bl = x_minor[list(min_bl_ind)]
        # 去除离群点
        if self.k3 != '-1':
            kd_all = KDTree(np.concatenate([x_major, x_minor]))

            def judge(indexes, min_start):
                count = 0
                for i in range(1, indexes.shape[0]):
                    if indexes[i] >= min_start:
                        count += 1
                return count == 1
            tmp = []
            for i in range(min_bl.shape[0]):
                tmp.append(judge(kd_all.query(min_bl[i].reshape(1, -1), k=self.k3, return_distance=False).reshape(-1), x_major.shape[0]))
            min_bl = min_bl[tmp]
        # 生成样本列表
        x_gen = []
        count = 0
        # 考察变异程度
        gen_extent = []
        # 生成样本使得两类样本数量相同
        for _ in range(int((num_major * self.r - num_minor) // 2)):
            # 随机选择少类样本中两个样本
            a, b = min_bl[random.randint(0, min_bl.shape[0] - 1)], min_bl[random.randint(0, min_bl.shape[0] - 1)]
            # 初始化空向量
            ap, bp = np.empty(n_features), np.empty(n_features)
            # 处理向量的每个特征
            for j in range(n_features):
                # 交叉的概率
                p_cross = math.exp(-(self.b + self.w * indexes.index(j)))
                if random.random() < p_cross:
                    # 交叉
                    ap[j], bp[j] = b[j], a[j]
                else:
                    # 变异
                    ap[j], bp[j] = a[j] - random.random() * (a[j] - b[j]), b[j] + random.random() * (a[j] - b[j])
                    # r1, r2 = random.random(), random.random()
                    # ap[j], bp[j] = r1 * a[j] + (1 - r1) * b[j], r2 * a[j] + (1 - r2) * b[j]
            x_gen.append(ap)
            x_gen.append(bp)
            gen_extent.append([a, b, ap, bp])
            count += 2
            print('\r>>> SMGO Processing: {} / {}'.format(count, (num_major - num_minor) // 2 * 2), end='')
        print('\r', end='')
        # 拼接少量样本、生成样本、大量样本
        x_new = np.concatenate([x_minor, x_major, np.array(x_gen)])
        # 拼接样本标签
        y_new = np.concatenate([np.array([label for _ in range(num_minor)]), np.array([1 - label for _ in range(num_major)]), np.array([label for _ in range(+ len(x_gen))])])
        return x_new, y_new


def stat(y):
    label = 1 if y[y == 0].shape[0] >= y[y == 1].shape[0] else 0  #
    num_minor, num_major = y[y == label].shape[0], y[y != label].shape[0]
    print('minor: {}, major: {}'.format(num_minor, num_major))


def visual2D(x, y):
    y = y.reshape(x.shape[0], 1)
    print(x.shape, y.shape)
    cc = np.concatenate((x, y), axis=1)
    print(cc[:5])
    ts = TSNE(n_components=2, random_state=33)
    ts.fit_transform(cc[:, :-1])
    em = ts.embedding_
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(em[:, 0], em[:, 1], c=cc[:, -1], cmap=plt.cm.Spectral)


def visual3D(x, y):
    y = y.reshape(x.shape[0], 1)
    print(x.shape, y.shape)
    cc = np.concatenate((x, y), axis=1)
    print(cc[:5])
    ts = TSNE(n_components=3, random_state=33)
    ts.fit_transform(cc[:, :-1])
    em = ts.embedding_
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(em[:, 0], em[:, 1], em[:, 2], c=cc[:, -1], cmap=plt.cm.Spectral)
