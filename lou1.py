import itertools
import os
import random

import networkx as nx
import numpy as np
from community import community_louvain
from igraph import VertexClustering
from sklearn.cluster import SpectralClustering, estimate_bandwidth
import matplotlib.pyplot as plt
import re
from collections import defaultdict, Counter
from math import pi, cos, sin

import igraph as ig

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def parse_interaction_file(file_path,k):
    edges = []
    in_and_interactions = False

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if line == "---------- AND Interactions ----------":
                in_and_interactions = True
                continue
            if line == "---------- OR Interactions ----------":
                break

            if in_and_interactions and line.startswith("I("):
                match = re.match(r"I\(([^)]*)\):\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", line)
                if match:
                    nodes_str = match.group(1)
                    weight = float(match.group(2))

                    nodes = list(nodes_str)
                    if len(nodes) >= 2 and len(nodes) <= k:
                        pairs = list(itertools.combinations(nodes, 2))
                        split_weight = weight / len(pairs)

                        for edge_nodes in pairs:
                            edge = tuple(sorted(edge_nodes))
                            found = False
                            for i, (e_node1, e_node2, e_weight) in enumerate(edges):
                                if (e_node1, e_node2) == edge:
                                    edges[i] = (e_node1, e_node2, e_weight + split_weight)
                                    found = True
                                    break
                            if not found :
                                edges.append((edge[0], edge[1], split_weight))

    print(f"解析完成，共找到 {len(edges)} 个有效连接")
    return edges


def create_similarity_matrix(edges):
    nodes = sorted(set().union(*[(e[0], e[1]) for e in edges]))
    n_nodes = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    W = np.zeros((n_nodes, n_nodes))
    # weights = [abs(w) for _, _, w in edges]
    # sigma = estimate_bandwidth(np.array(weights).reshape(-1, 1))

    for node1, node2, weight in edges:
        i = node_to_idx[node1]
        j = node_to_idx[node2]
        # 使用高斯核函数转换权重为相似度
        W[i, j] = abs(weight)  ####np.exp(-(abs(weight)**2 / (2 * sigma**2)))
        W[j, i] = W[i, j]

    return W, nodes

def main():
    import argparse
    parser = argparse.ArgumentParser(description='交互网络分析与可视化')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='聚类数量（默认: %(default)s）')
    parser.add_argument('--louk', type=int, default=999,
                        help='阶数（默认: %(default)s）')
    parser.add_argument('--i', type=int, default=0,
                        help='样本编号（默认: %(default)s）')
    parser.add_argument('--datak', type=int, default=-1,
                        help='result阶数（默认: %(default)s）')
    args = parser.parse_args()

    # default_path = "./results/20250422_examine/result/dataset=custom-natcomp-for-bert-2025-5-test_model=Bert-base#pretrain_seed=0/players=players-manual_k=3_mode=pq_lbl=predict_baseline=unk_bg=ori#_loss=l1_optim=sgd_autolr=v1_mom=0.999_niters=50000_qcoef=0.05/data/sample0/interaction.txt"
    default_path = f"./results/20250331_examine/result/dataset=custom-imdb-for-bertweet-nips2024-ucb-test_model=BERTweet#pretrain_seed=0/players=players-manual_k={args.datak}_mode=pq_lbl=correct_baseline=unk_bg=ori#_loss=l1_optim=sgd_autolr=v1_mom=0.999_niters=50000_qcoef=0.05/data/sample{args.i}/interaction.txt"
    # default_path = f"./results/20250331_examine/result/dataset=custom-imdb-for-bertweet-2-test_model=BERTweet#pretrain_seed=0/players=players-manual_mode=pq_lbl=correct_baseline=unk_bg=ori#_loss=l1_optim=sgd_autolr=v1_mom=0.999_niters=50000_qcoef=0.05/data/sample{args.i}/interaction.txt"

    edges = parse_interaction_file(default_path, args.louk)

    edges = [(n1, n2, w) for n1, n2, w in edges]

    W, nodes = create_similarity_matrix(edges)

    G = nx.Graph()
    for i, node in enumerate(nodes):
        G.add_node(node)
    for n1, n2, w in edges:
        G.add_edge(n1, n2,original_weight=w, weight=abs(w))

    igG = ig.Graph.from_networkx(G)

    weight_scale = 1.0  # 初始权重缩放因子
    step = 0.1  # 每次调整权重的步长
    max_attempts = 100
    target_clusters = args.n_clusters
    attempts = 0

    while attempts < max_attempts:
        if "weight" in igG.es.attributes():
            igG.es["weight"] = [w * weight_scale for w in igG.es["weight"]]
        else:
            igG.es["weight"] = [1.0 * weight_scale] * igG.ecount()

        partition = igG.community_multilevel(weights="weight")
        labels = list(partition.membership)
        n_clusters = len(set(labels))

        if n_clusters == target_clusters:
            print(f"找到目标社区数量 {target_clusters}，使用的权重缩放因子为 {weight_scale}")
            break

        if n_clusters > target_clusters:
            weight_scale += step
        else:
            weight_scale -= step

        weight_scale = max(weight_scale, 0.01)

        attempts += 1
        print(f"尝试 {attempts}: 当前权重缩放因子为 {weight_scale}, 社区数量为 {n_clusters}")

    if attempts == max_attempts:
        raise ValueError(f"在 {max_attempts} 次尝试后未能找到目标社区数量 {target_clusters}")

    # custom_labels = {}
    # for node in ['A', 'B' ]:
    #     custom_labels[node] = 1
    # for node in ['C', 'D', 'E']:
    #     custom_labels[node] = 0
    # # 默认社区为 2
    # membership = [custom_labels.get(node, 2) for node in nodes]
    # partition = VertexClustering(igG, membership)
    # labels = membership
    # n_clusters = len(set(labels))

    with open(default_path, 'r', encoding='utf-8') as file:
        text_lines = file.read()
    node_to_word = {}
    for line in text_lines.splitlines():
        if line.strip().startswith('Player'):
            parts = line.split(':')
            if len(parts) >= 2:
                node = parts[0].replace('Player', '').strip()
                word = parts[1].strip()
                node_to_word[node] = word
    # label_dict = {node: node_to_word[node] for node in G.nodes()}
    label_dict = {node: f"{node}:{node_to_word[node]}" for node in G.nodes()}

    node_color_map = {
        0: '#a1d99b', 1: '#fdae6b', 2: '#a2d2ff',
        3: '#e34a33', 4: '#9e9ac8', 5: '#d94801',
        6: '#f1b6da', 7: '#cccccc', 8: '#ffefc3', 9: '#a8ddb5'
    }

    # 按社区分组节点
    community_nodes = defaultdict(list)
    for node, label in zip(G.nodes(), labels):
        community_nodes[label].append(node)
    if len(community_nodes) < args.n_clusters:
        print(f"警告: 只有 {len(community_nodes)} 个社区有节点，调整聚类数为 {len(community_nodes)}")
        n_clusters = len(community_nodes)

    plt.figure(figsize=(16, 9))

    # 创建子图
    grid_spacing = 3.0
    cell_size = 2.0
    grid_positions = {
        0: (-1, 1),
        1: (0, 1),
        2: (1, 1),
        3: (-1, 0),
        # 4: (0, 0),    # 中心(留空)
        5: (1, 0),
        6: (-1, -1),
        7: (0, -1),
        8: (1, -1)
    }

    # 为社区分配子图
    pos = {}
    community_centers = {}
    for comm_id in range(n_clusters):
        if comm_id not in community_nodes or not community_nodes[comm_id]:
            continue
        grid_idx = comm_id if comm_id < 4 else comm_id + 1
        grid_x, grid_y = grid_positions[grid_idx]
        nodes_in_comm = community_nodes[comm_id]
        subgraph = G.subgraph(nodes_in_comm)
        sub_pos = nx.spring_layout(subgraph, seed=42, weight='weight',
                                   k=0.15 * cell_size, iterations=50)
        for node, (x, y) in sub_pos.items():
            pos[node] = (grid_x * grid_spacing + x * cell_size,
                         grid_y * grid_spacing + y * cell_size)
        community_centers[comm_id] = (grid_x * grid_spacing, grid_y * grid_spacing)
    missing_nodes = [node for node in G.nodes() if node not in pos]
    if missing_nodes:
        print(f"警告: {len(missing_nodes)} 个节点没有分配到任何社区，将为它们分配默认位置")
        center_x, center_y = 0, 0
        angle_step = 2 * pi / len(missing_nodes)
        for i, node in enumerate(missing_nodes):
            angle = i * angle_step
            pos[node] = (center_x + 0.5 * cos(angle), center_y + 0.5 * sin(angle))

    # 节点背景
    intra_community_edges = []
    inter_community_edges = []
    for u, v in G.edges():
        u_label = labels[list(G.nodes()).index(u)]
        v_label = labels[list(G.nodes()).index(v)]
        if u_label == v_label:
            intra_community_edges.append((u, v))
        else:
            inter_community_edges.append((u, v))
    node_to_label = dict(zip(G.nodes(), labels))
    node_colors = [node_color_map[node_to_label[node]] for node in G.nodes()]

    # nx.draw_networkx_nodes(
    #     G, pos,
    #     node_color=node_colors,
    #     node_size=7800,
    #     alpha=0.8
    # )
    # # 绘制节点标签
    # nx.draw_networkx_labels(
    #     G, pos,
    #     labels=label_dict,
    #     font_size=28,
    #     font_color='black'
    # )

    ax = plt.gca()
    fig = plt.gcf()

    for i, (node, (x, y)) in enumerate(pos.items()):
        label = label_dict[node]
        text = ax.text(
            x, y, label,
            ha='center', va='center',
            fontsize=20,
            zorder=2
        )
        # 获取文字边界框
        renderer = fig.canvas.get_renderer()
        bbox = text.get_window_extent(renderer=renderer)
        bbox_data = bbox.transformed(ax.transData.inverted())

        width = bbox_data.width
        height = bbox_data.height

        pad1 = 0.5  # 控制框的边缘留白
        pad2 = 0.1
        box = FancyBboxPatch(
            (x - width / 2 - pad1, y - 2 * height / 2 - pad2),
            width + 2 * pad1,
            height + 2 * pad2,
            boxstyle="round,pad=0.02",  # 可改为 "square" 或 "round"
            linewidth=1,
            edgecolor='white',
            facecolor=node_color_map[node_to_label[node]],
            alpha=0.8,
            zorder=1
        )
        ax.add_patch(box)

    print("实际用到的社区标签:", set(labels))
    print("每个标签的节点数:", Counter(labels))

    # 边的格式设置
    max_abs_weight = max(abs(w) for _, _, w in edges)
    edge_colors = []
    edge_widths = []
    edge_styles = []
    edge_labels_dict = {}
    for u, v in G.edges():
        data = G.get_edge_data(u, v)
        node1_label = labels[list(G.nodes()).index(u)]
        node2_label = labels[list(G.nodes()).index(v)]
        original_weight = data['original_weight']
        # edge_colors.append('#d62728' if original_weight > 0 else '#2ca02c')
        edge_colors.append('#ff6b6b' if original_weight > 0 else '#7ed38b')
        edge_widths.append(4.5 + 6.5 * abs(original_weight) / max_abs_weight)
        edge_styles.append('solid' if node1_label == node2_label else 'dashed')
        edge_labels_dict[(u, v)] = f"{original_weight:.2f}"

    # 绘制边 - 按线型分组绘制更高效
    solid_edges = [(u, v) for (u, v), style in zip(G.edges(), edge_styles) if style == 'solid']
    dashed_edges = [(u, v) for (u, v), style in zip(G.edges(), edge_styles) if style == 'dashed']

    # 绘制实线边
    nx.draw_networkx_edges(
        G, pos,
        edgelist=solid_edges,
        edge_color=[edge_colors[i] for i, (u, v) in enumerate(G.edges()) if (u, v) in solid_edges],
        width=[edge_widths[i] for i, (u, v) in enumerate(G.edges()) if (u, v) in solid_edges],
        style='solid',
        alpha=0.9
    )

    # 绘制虚线边
    nx.draw_networkx_edges(
        G, pos,
        edgelist=dashed_edges,
        edge_color=[edge_colors[i] for i, (u, v) in enumerate(G.edges()) if (u, v) in dashed_edges],
        width=[edge_widths[i] for i, (u, v) in enumerate(G.edges()) if (u, v) in dashed_edges],
        style='dashed',
        alpha=0.3
    )

    # 边上的交互值 - 只绘制绝对值≥0.1的标签
    filtered_edge_labels = {
        edge: label for edge, label in edge_labels_dict.items()
        if abs(float(label)) > 0.5
    }

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=filtered_edge_labels,  # 只传入过滤后的标签
        font_color='black',
        font_size=14,
        rotate=False,
        horizontalalignment='center',
        verticalalignment='center',
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="none",
            alpha=0.7
        )
    )

    match_dataset = re.search(r"dataset=([^_]+)", default_path)
    match_players = re.search(r"players=([^_]+)", default_path)
    match_sample = re.search(r"/sample(\d+)/", default_path)
    dataset = match_dataset.group(1) if match_dataset else "unknown_dataset"
    players = match_players.group(1) if match_players else "unknown_players"
    sample = match_sample.group(1) if match_sample else "unknown_sample"

    output_dir = './visual/visualizations/lou'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"./visual/visualizations/lou/datak={args.datak}_louk={args.louk}", exist_ok=True)
    output_filename = f"./datak={args.datak}_louk={args.louk}/{dataset}_{players}_sample_{sample}_n={n_clusters}_0508.png"
    output_filepath = os.path.join(output_dir, output_filename)
    plt.savefig(output_filepath)
    plt.box(False)
    plt.axis('off')
    plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    main()