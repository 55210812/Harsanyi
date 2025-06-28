import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
 
def load_data(filename):
    """
    载入数据
    :param filename: 文件名
    :return: numpy array 格式的数据
    """
    data = np.loadtxt(filename, delimiter='\t')
    return data
 
def distance(x1, x2):
    """
    获得两个样本点之间的欧几里得距离
    :param x1: 样本点1
    :param x2: 样本点2
    :return: 两个样本点之间的距离
    """
    return np.linalg.norm(x1 - x2)
 
def get_dist_matrix(data):
    """
    获取距离矩阵
    :param data: 样本集合
    :return: 距离矩阵
    """
    return np.linalg.norm(data[:, np.newaxis] - data[np.newaxis, :], axis=-1)
 
def getW(data, k):
    """
    获得邻接矩阵 W
    :param data: 样本集合
    :param k: KNN参数
    :return: 邻接矩阵 W
    """
    n = len(data)
    dist_matrix = get_dist_matrix(data)
    W = np.zeros((n, n))
 
    for idx in range(n):
        # 获取最近k个邻居的索引
        idx_array = np.argsort(dist_matrix[idx])[1:k+1]  # 跳过自己
        W[idx, idx_array] = 1
    
    # 确保邻接矩阵是对称的
    return (W + W.T) / 2
 
def getD(W):
    """
    获得度矩阵
    :param W: 邻接矩阵
    :return: 度矩阵 D
    """
    return np.diag(np.sum(W, axis=1))
 
def getL(D, W):
    """
    获得拉普拉斯矩阵
    :param D: 度矩阵
    :param W: 邻接矩阵
    :return: 拉普拉斯矩阵 L
    """
    return D - W
 
def getEigen(L, cluster_num):
    """
    获得拉普拉斯矩阵的特征向量
    :param L: 拉普拉斯矩阵
    :param cluster_num: 聚类数目
    :return: 选定特征值对应的特征向量
    """
    eigval, eigvec = np.linalg.eig(L)
    ix = np.argsort(eigval)[:cluster_num]  # 选择最小的cluster_num个特征值的索引
    return eigvec[:, ix]
 
def plotRes(data, clusterResult, clusterNum):
    """
    结果可视化
    :param data: 样本集
    :param clusterResult: 聚类结果
    :param clusterNum: 聚类个数
    """
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange']
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        plt.scatter(data[clusterResult == i, 0], data[clusterResult == i, 1], c=color, marker='+')
    
    plt.title(f'Clustering Result with {clusterNum} clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
 
def cluster(data, cluster_num, k):
    """
    聚类函数
    :param data: 输入数据
    :param cluster_num: 聚类数目
    :param k: KNN参数
    :return: 聚类标签
    """
    W = getW(data, k)
    D = getD(W)
    L = getL(D, W)
    eigvec = getEigen(L, cluster_num)
    
    # 使用KMeans进行聚类
    clf = KMeans(n_clusters=cluster_num)
    label = clf.fit_predict(eigvec)  # 直接使用fit_predict
    return label
 
if __name__ == '__main__':
    cluster_num = 7
    knn_k = 5
    filename = '../data/Aggregation_cluster=7.txt'
    
    data = load_data(filename=filename)
    data = data[:, :-1]  # 去除最后一列（假设为标签列）
    
    label = cluster(data, cluster_num, knn_k)
    plotRes(data, label, cluster_num)