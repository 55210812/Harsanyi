import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import Union, List, Dict

class SpectralClustering:
    def __init__(self, n_clusters: int = 2, knn_k: int = 5):
        """
        初始化谱聚类器
        :param n_clusters: 聚类数量
        :param knn_k: KNN参数
        """
        self.n_clusters = n_clusters
        self.knn_k = knn_k
        self.labels_ = None

    @staticmethod
    def _get_dist_matrix(data: np.ndarray) -> np.ndarray:
        """
        计算距离矩阵
        :param data: 样本数据 (n_samples, n_features)
        :return: 距离矩阵 (n_samples, n_samples)
        """
        # 添加小常数防止数值不稳定
        return np.linalg.norm(data[:, np.newaxis] - data[np.newaxis, :], axis=-1) + 1e-10

    @staticmethod
    def _get_adjacency_matrix(dist_matrix: np.ndarray, k: int) -> np.ndarray:
        """
        构建邻接矩阵 (KNN方法)
        :param dist_matrix: 距离矩阵
        :param k: KNN参数
        :return: 邻接矩阵
        """
        n = len(dist_matrix)
        W = np.zeros((n, n))
        
        # 自适应KNN参数
        k = min(k, n-1)  # 确保k不超过节点数
        
        for idx in range(n):
            # 使用高斯核权重而不是0/1权重
            sorted_idx = np.argsort(dist_matrix[idx])[1:k+1]  # 跳过自己
            weights = np.exp(-dist_matrix[idx, sorted_idx]**2 / (2 * np.median(dist_matrix)**2))
            W[idx, sorted_idx] = weights
        
        # 确保对称并归一化
        W = (W + W.T) / 2
        return W / np.max(W)  # 归一化到[0,1]

    @staticmethod
    def _get_laplacian(W: np.ndarray) -> np.ndarray:
        """
        计算归一化拉普拉斯矩阵
        :param W: 邻接矩阵
        :return: 归一化拉普拉斯矩阵
        """
        D = np.diag(np.sum(W, axis=1))
        # 处理孤立节点（度为0）
        diag = np.diag(D).copy()  # 创建可修改的副本
        diag[diag == 0] = 1e-10  # 添加小常数避免除零
        D_inv_sqrt = np.diag(1.0 / np.sqrt(diag))
        return np.eye(len(W)) - D_inv_sqrt @ W @ D_inv_sqrt

    @staticmethod
    def _get_eigenvectors(L: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        计算特征向量
        :param L: 拉普拉斯矩阵
        :param n_clusters: 需要的特征向量数量
        :return: 特征向量矩阵 (n_samples, n_clusters)
        """
        # 使用更稳定的eigh计算对称矩阵特征值
        eigval, eigvec = np.linalg.eigh(L)
        
        # 选择前n_clusters个最小的非零特征值对应的特征向量
        ix = np.argsort(eigval)[1:n_clusters+1]  # 跳过第一个0特征值
        
        # 归一化特征向量并处理NaN
        eigvec = eigvec[:, ix]
        norms = np.linalg.norm(eigvec, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10  # 防止除零
        eigvec = eigvec / norms
        
        # 处理可能的NaN值
        eigvec = np.nan_to_num(eigvec, nan=0.0)
        
        return eigvec

    def fit(self, X: Union[np.ndarray, nx.Graph]) -> 'SpectralClustering':
        """
        拟合数据
        :param X: 输入数据，可以是numpy数组或NetworkX图
        :return: self
        """
        if isinstance(X, nx.Graph):
            # 从图数据构建邻接矩阵
            W = nx.to_numpy_array(X)
            print(W)
        else:
            # 从特征数据构建邻接矩阵
            dist_matrix = self._get_dist_matrix(X)
            W = self._get_adjacency_matrix(dist_matrix, self.knn_k)

        L = self._get_laplacian(W)
        eigvec = self._get_eigenvectors(L, self.n_clusters)
        
        # 使用KMeans聚类特征向量（显式设置n_init避免警告）
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        self.labels_ = kmeans.fit_predict(eigvec)
        return self

    def fit_predict(self, X: Union[np.ndarray, nx.Graph]) -> np.ndarray:
        """
        拟合数据并返回聚类标签
        :param X: 输入数据
        :return: 聚类标签
        """
        return self.fit(X).labels_

    @staticmethod
    def visualize_clusters(
            X: Union[np.ndarray, nx.Graph],
            labels: np.ndarray,
            node_names: Dict[str, str] = None,
            edge_weights: bool = False,
            title: str = "Spectral Clustering Result",
            output_file: str = None,
            dpi: int = 300
        ) -> None:
        """
        可视化聚类结果
        :param X: 输入数据
        :param labels: 聚类标签
        :param node_names: 节点名称映射
        :param edge_weights: 是否显示边权重
        :param title: 图表标题
        """
        plt.figure(figsize=(10, 8))
        
        if isinstance(X, nx.Graph):
            # 可视化网络图 - 使用更稳定的布局算法
            pos = nx.kamada_kawai_layout(X, scale=2)
            colors = plt.cm.tab20(np.linspace(0, 1, len(set(labels))))
            
            # 绘制节点 - 增加节点大小和间距
            for i, label in enumerate(set(labels)):
                nodes = [node for node, l in zip(X.nodes(), labels) if l == label]
                nx.draw_networkx_nodes(
                    X, pos, nodelist=nodes,
                    node_color=[colors[i]] * len(nodes),
                    node_size=800,
                    alpha=0.9,
                    label=f'Cluster {label}'
                )
            
            # 绘制边 - 增加边宽度
            edges = nx.draw_networkx_edges(
                X, pos,
                width=2.0,
                alpha=0.7,
                edge_color='gray'
            )
            
            # 绘制边权重 - 优化显示
            if edge_weights:
                edge_labels = {
                    (u, v): f"{d['weight']:.2f}"
                    for u, v, d in X.edges(data=True)
                    if d['weight'] > 0.1  # 只显示显著权重
                }
                nx.draw_networkx_edge_labels(
                    X, pos,
                    edge_labels=edge_labels,
                    font_color='red',
                    font_size=9,
                    bbox=dict(alpha=0.8)
                )
            
            # 绘制节点标签 - 优化显示
            if node_names:
                nx.draw_networkx_labels(
                    X, pos, labels=node_names,
                    font_size=9,
                    font_weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7))
            else:
                nx.draw_networkx_labels(
                    X, pos,
                    font_size=9,
                    font_weight='bold')
            
            # 添加图例
            plt.legend(scatterpoints=1, framealpha=0.5)
        
        else:
            # 可视化特征数据
            scatter_colors = plt.cm.tab10(np.linspace(0, 1, len(set(labels))))
            
            for i, label in enumerate(set(labels)):
                plt.scatter(
                    X[labels == label, 0], X[labels == label, 1],
                    c=[scatter_colors[i]], marker='o',
                    label=f'Cluster {label}'
                )
            
            plt.legend()
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        if output_file:
            import os
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.show()

def load_interaction_graph(filepath: str) -> nx.Graph:
    """从interaction.txt文件加载网络图"""
    G = nx.Graph()
    current_section = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('----------'):
                # 切换section
                if 'Players' in line:
                    current_section = 'players'
                elif 'AND Interactions' in line:
                    current_section = 'and_interactions'
                elif 'OR Interactions' in line:
                    current_section = 'or_interactions'
                continue
                
            if current_section == 'players':
                # 解析玩家节点 - 处理多种格式：
                line = line.strip()
                if not line:
                    continue
                    
                # 尝试分割玩家名和描述
                if ':' in line:
                    parts = line.split(':', 1)
                    player = parts[0].strip()
                    desc = parts[1].strip() if len(parts) > 1 else ''
                else:
                    player = line.strip()
                    desc = ''
                
                # 确保节点存在且有description属性
                if player:  # 确保玩家名不为空
                    if player not in G:
                        G.add_node(player, description=desc)
                    elif 'description' not in G.nodes[player]:
                        G.nodes[player]['description'] = desc
                    
            elif current_section == 'and_interactions':
                # 解析AND交互边权
                if line.startswith('I('):
                    try:
                        # 使用正则表达式提取数值部分
                        import re
                        match = re.search(r'I\(([A-Za-z]+)\)\s*[:=]?\s*([-+]?\d*\.?\d+)', line)
                        if match:
                            players = match.group(1)
                            weight = float(match.group(2))
                            if len(players) >= 2:
                                G.add_edge(players[0], players[1], weight=abs(weight))
                                print(f"Graph edge {players[0]}-{players[1]}: {G[players[0]][players[1]]['weight']}")
                        else:
                            print(f"跳过无法解析的行: {line}")
                    except Exception as e:
                        print(f"解析行时出错: {line} - {str(e)}")
    
    return G

if __name__ == '__main__':
    # from sklearn.datasets import make_blobs
    # X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    
    # sc = SpectralClustering(n_clusters=4)
    # labels = sc.fit_predict(X)
    # SpectralClustering.visualize_clusters(
    #     X, labels, 
    #     title="Spectral Clustering on Synthetic Data",
    #     output_file="output/synthetic_data_clustering.png"
    # )
    
    # G = nx.karate_club_graph()
    # node_names = {i: f"P{i}" for i in G.nodes()}
    
    # sc = SpectralClustering(n_clusters=2)
    # labels = sc.fit_predict(G)
    # SpectralClustering.visualize_clusters(
    #     G, labels, 
    #     node_names=node_names,
    #     edge_weights=True,
    #     title="Spectral Clustering on Zachary's Karate Club",
    #     output_file="output/karate_club_clustering.png"
    # )

    interaction_file = "/mnt/data/hqdeng7/CSCIENCE/interaction_nlp_harsanyi/results/20250410_harsanyi_all_optimization_0.025/result/dataset=custom-imdb-for-bertweet-nips2024-ucb-test_model=BERTweet#pretrain_seed=0/players=players-all_lbl=correct_baseline=unk_bg=ori#/data/sample2/interaction.txt"
    try:
        G = load_interaction_graph(interaction_file)

        sc = SpectralClustering(n_clusters=3)  # 根据需求调整聚类数量
        labels = sc.fit_predict(G)
        
        # 创建节点名称映射，处理可能缺失的description属性
        node_names = {
            n: f"{n}: {G.nodes[n].get('description', '')}" 
            for n in G.nodes()
        }
        
        SpectralClustering.visualize_clusters(
            G, labels,
            node_names=node_names,
            edge_weights=True,
            title="Spectral Clustering on IMDB Interaction Data",
            output_file="output/imdb_interaction_clustering2.png"
        )
    except FileNotFoundError:
        print(f"文件未找到: {interaction_file}")
    except Exception as e:
        print(f"处理文件时出错: {e}")
