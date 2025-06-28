import argparse
import networkx as nx
import os
import re
import matplotlib.pyplot as plt
import math
from sklearn.cluster import AffinityPropagation
from networkx.algorithms.community import louvain_communities, greedy_modularity_communities
from girvan_newman import girvan_newman
import itertools
import random

import numpy as np
from sklearn.cluster import SpectralClustering, estimate_bandwidth
import matplotlib.pyplot as plt
import re
from collections import defaultdict, Counter
from math import pi, cos, sin

import igraph as ig

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Community detection from interaction data')
    parser.add_argument('--file_path', required=True, help='Path to interaction.txt file')
    parser.add_argument('--output', help='Output PNG file name')
    parser.add_argument('--function', default="louvain", choices=["louvain", "gw", "AP"], 
                       help='Community detection function (default: louvain)')
    
    args = parser.parse_args()

    # 读取文件
    with open(args.file_path, 'r') as f:
        data = f.read()

    # 解析玩家信息
    players = {}
    # 原解析逻辑(保留)
    player_section = re.search(r'---------- Players ----------(.*?)---------- AND Interactions \(Pairwise Only\)', data, re.DOTALL)
    if player_section:
        for line in player_section.group(1).strip().split('\n'):
            match = re.match(r'Player (\w+): (.*)', line)
            if match:
                players[match.group(1)] = match.group(2)
    
    # 新解析逻辑(处理像素块格式)
    # 匹配格式: "Player 0: block_0_0"
    # player_section = re.search(r'Player \d+: block_\d+_\d+', data)
    # if player_section:
    #     #print("Player section found")
    #     for line in data.split('\n'):
    #         if line.startswith('Player'):
    #             match = re.match(r'Player (\d+): (block_\d+_\d+)', line)
    #             if match:
    #                 players[match.group(1)] = match.group(2)
                    #print(f"Player {match.group(1)}: {match.group(2)}")

    # 解析AND Interactions
    G = nx.Graph()
    # 原解析逻辑(保留)
    interaction_section = re.search(r'---------- AND Interactions.*?---------- OR Interactions', data, re.DOTALL)
    if interaction_section:
        for line in interaction_section.group(0).split('\n'):
            match = re.match(r'I\((\w+)\): ([+-]?\d*\.\d+)', line)
            if match:
                nodes = list(match.group(1))
                weight = float(match.group(2))
                abs_weight = abs(weight)
                if weight > 0:
                    weight = 0
                G.add_edge(nodes[0], nodes[1], weight=weight, abs_weight=abs_weight)
    
    # 新解析逻辑(处理像素块交互格式)
    # 匹配格式: "I(148,149): 6.037294     ([block_9_4][block_9_5])"
    # for line in data.split('\n'):
    #     if line.startswith('I('):
    #         #print("Interaction section found")
    #         match = re.match(r'I\((\d+),(\d+)\): ([+-]?\d*\.\d+)\s+\(\[block_\d+_\d+\]\[block_\d+_\d+\]\)', line)
    #         if match:
    #             #print(f"Match found: {match.group(0)}")
    #             node1, node2 = match.group(1), match.group(2)
    #             weight = float(match.group(3))
    #             #print(f"Adding edge: {node1}, {node2}, weight: {weight}")
    #             abs_weight = abs(weight)
    #             if weight > 0:
    #                 weight = 0
    #             G.add_edge(node1, node2, weight=weight, abs_weight=abs_weight)
    # interaction_section = re.search(r'---------- AND Interactions \(Pairwise Only\) ----------(.*?)(?:Sum: |\Z)', data, re.DOTALL)
    # if interaction_section:
    #     seen_edges = set()
    #     for line in interaction_section.group(1).strip().split('\n'):
    #         match = re.match(r'I\((\w+)\): ([+-]?\d*\.\d+)', line)
    #         if match:
    #             edge = match.group(1)
    #             if edge not in seen_edges:  # Skip duplicates
    #                 seen_edges.add(edge)
    #                 nodes = list(edge)
    #                 weight = float(match.group(2))
    #                 abs_weight = abs(weight)
    #                 G.add_edge(nodes[0], nodes[1], weight=weight, abs_weight=abs_weight)

    # 选择社区检测算法
    print(G.edges())
    if args.function == "louvain":
        if len(G.edges()) == 0:
            print("Warning: Graph has no edges, returning empty communities")
            communities = []
        else:
            # Check if all abs_weights are zero
            all_zero = all(data['abs_weight'] == 0 for _, _, data in G.edges(data=True))
            if all_zero:
                print("Warning: All edge weights are zero, returning empty communities")
                communities = []
            else:
                # 偏好更小的社区
                # communities = louvain_communities(G, resolution=1.8)

                # # 偏好更大的社区
                # communities = louvain_communities(G, resolution=1.8)

                # # 使用流量作为权重
                # communities = louvain_communities(G, weight='traffic')

                # # 更严格的收敛条件
                # communities = louvain_communities(G, threshold=1e-6)
                #communities = list(louvain_communities(G, weight='abs_weight', resolution=1))
                #communities = list(louvain_communities(G, weight='abs_weight', resolution=1.8))
                #communities = list(louvain_communities(G, weight='abs_weight', resolution=1.8))
                communities = list(louvain_communities(G, weight='weight', resolution=1.3))
    elif args.function == "gw":
        communities = list(girvan_newman(G))
    elif args.function == "AP":
        af = AffinityPropagation(preference=-50, random_state=0).fit(G)
        cluster_centers_indices = af.cluster_centers_indices_
        n_clusters_ = len(cluster_centers_indices)

    # 转换社区为可哈希类型
    communities = [frozenset(community) for community in communities]
    
    # 打印社区结果
    output_path = f"/mnt/data/hqdeng7/CSCIENCE/interaction_nlp_harsanyi/group/louvain_1.3_-_weight/{args.output}.txt"
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    print(f"{args.function}社区检测结果:")
    with open (output_path, 'w') as f:
        f.write(f"{args.function}社区检测结果:\n")
        for i, community in enumerate(communities):
            # 计算社区内平均边权
            subgraph = nx.subgraph(G, community)
            edge_weights = [data['abs_weight'] for _, _, data in subgraph.edges(data=True)]
            avg_weight = sum(edge_weights)/len(edge_weights) if edge_weights else 0
            
            print(f"社区 {i+1}: {[players[n] for n in community]}, 平均边权: {avg_weight:.4f}")
            f.write(f"社区 {i+1}: {[n for n in community]}: {[players[n] for n in community]}: 平均边权 {avg_weight:.4f}\n")


    # # 可视化设置
    # node_color_map = {
    #     0: '#a1d99b', 1: '#fdae6b', 2: '#a2d2ff',
    #     3: '#e34a33', 4: '#9e9ac8', 5: '#d94801',
    #     6: '#f1b6da', 7: '#cccccc', 8: '#ffefc3', 9: '#a8ddb5'
    # }

    # # 按社区分组节点
    # community_nodes = defaultdict(list)
    # for i, community in enumerate(communities):
    #     for node in community:
    #         community_nodes[i].append(node)
    # n_clusters = len(community_nodes)

    # plt.figure(figsize=(16, 9))

    # # 创建子图布局
    # grid_spacing = 3.0
    # cell_size = 2.0
    # grid_positions = {
    #     0: (-1, 1),
    #     1: (0, 1),
    #     2: (1, 1),
    #     3: (-1, 0),
    #     5: (1, 0),
    #     6: (-1, -1),
    #     7: (0, -1),
    #     8: (1, -1)
    # }

    # # 为社区分配子图位置
    # pos = {}
    # community_centers = {}
    # for comm_id in range(n_clusters):
    #     if comm_id not in community_nodes or not community_nodes[comm_id]:
    #         continue
    #     grid_idx = comm_id if comm_id < 4 else comm_id + 1
    #     grid_x, grid_y = grid_positions[grid_idx]
    #     nodes_in_comm = community_nodes[comm_id]
    #     subgraph = G.subgraph(nodes_in_comm)
    #     sub_pos = nx.spring_layout(subgraph, seed=42, weight='abs_weight',
    #                              k=0.15 * cell_size, iterations=50)
    #     for node, (x, y) in sub_pos.items():
    #         pos[node] = (grid_x * grid_spacing + x * cell_size,
    #                      grid_y * grid_spacing + y * cell_size)
    #     community_centers[comm_id] = (grid_x * grid_spacing, grid_y * grid_spacing)
    # missing_nodes = [node for node in G.nodes() if node not in pos]
    # if missing_nodes:
    #     center_x, center_y = 0, 0
    #     angle_step = 2 * pi / len(missing_nodes)
    #     for i, node in enumerate(missing_nodes):
    #         angle = i * angle_step
    #         pos[node] = (center_x + 0.5 * cos(angle), center_y + 0.5 * sin(angle))

    # # 节点背景和标签
    # node_to_label = {node: comm_id for comm_id, nodes in community_nodes.items() for node in nodes}
    # node_colors = [node_color_map[node_to_label.get(node, 0)] for node in G.nodes()]
    # label_dict = {node: f"{node}:{players[node]}" for node in G.nodes()}

    # ax = plt.gca()
    # fig = plt.gcf()

    # for i, (node, (x, y)) in enumerate(pos.items()):
    #     label = label_dict[node]
    #     text = ax.text(
    #         x, y, label,
    #         ha='center', va='center',
    #         fontsize=20,
    #         zorder=2
    #     )
    #     # 获取文字边界框
    #     renderer = fig.canvas.get_renderer()
    #     bbox = text.get_window_extent(renderer=renderer)
    #     bbox_data = bbox.transformed(ax.transData.inverted())

    #     width = bbox_data.width
    #     height = bbox_data.height

    #     pad1 = 0.5  # 控制框的边缘留白
    #     pad2 = 0.1
    #     box = FancyBboxPatch(
    #         (x - width / 2 - pad1, y - 2 * height / 2 - pad2),
    #         width + 2 * pad1,
    #         height + 2 * pad2,
    #         boxstyle="round,pad=0.02",
    #         linewidth=1,
    #         edgecolor='white',
    #         facecolor=node_color_map[node_to_label.get(node, 0)],
    #         alpha=0.8,
    #         zorder=1
    #     )
    #     ax.add_patch(box)

    # # 边的格式设置
    # max_abs_weight = max(abs(data['weight']) for _, _, data in G.edges(data=True))
    # edge_colors = []
    # edge_widths = []
    # edge_styles = []
    # edge_labels_dict = {}
    # for u, v in G.edges():
    #     data = G.get_edge_data(u, v)
    #     node1_label = node_to_label.get(u, 0)
    #     node2_label = node_to_label.get(v, 0)
    #     original_weight = data['weight']
    #     edge_colors.append('#ff6b6b' if original_weight > 0 else '#7ed38b')
    #     edge_widths.append(4.5 + 6.5 * abs(original_weight) / max_abs_weight)
    #     edge_styles.append('solid' if node1_label == node2_label else 'dashed')
    #     edge_labels_dict[(u, v)] = f"{original_weight:.2f}"

    # # 绘制边 - 按线型分组绘制
    # solid_edges = [(u, v) for (u, v), style in zip(G.edges(), edge_styles) if style == 'solid']
    # dashed_edges = [(u, v) for (u, v), style in zip(G.edges(), edge_styles) if style == 'dashed']

    # # 绘制实线边
    # nx.draw_networkx_edges(
    #     G, pos,
    #     edgelist=solid_edges,
    #     edge_color=[edge_colors[i] for i, (u, v) in enumerate(G.edges()) if (u, v) in solid_edges],
    #     width=[edge_widths[i] for i, (u, v) in enumerate(G.edges()) if (u, v) in solid_edges],
    #     style='solid',
    #     alpha=0.9
    # )

    # # 绘制虚线边
    # nx.draw_networkx_edges(
    #     G, pos,
    #     edgelist=dashed_edges,
    #     edge_color=[edge_colors[i] for i, (u, v) in enumerate(G.edges()) if (u, v) in dashed_edges],
    #     width=[edge_widths[i] for i, (u, v) in enumerate(G.edges()) if (u, v) in dashed_edges],
    #     style='dashed',
    #     alpha=0.3
    # )

    # # 边上的交互值 - 只绘制绝对值≥0.5的标签
    # filtered_edge_labels = {
    #     edge: label for edge, label in edge_labels_dict.items()
    #     if abs(float(label)) > 0.5
    # }

    # nx.draw_networkx_edge_labels(
    #     G, pos,
    #     edge_labels=filtered_edge_labels,
    #     font_color='black',
    #     font_size=14,
    #     rotate=False,
    #     horizontalalignment='center',
    #     verticalalignment='center',
    #     bbox=dict(
    #         boxstyle="round",
    #         facecolor="white",
    #         edgecolor="none",
    #         alpha=0.7
    #     )
    # )

    # plt.box(False)
    # plt.axis('off')
    # plt.tight_layout()
        # 可视化
    weights = [abs(G[u][v]['weight'])*2 for u,v in G.edges()]
    edge_colors = ['red' if G[u][v]['weight'] < 0 else 'green' for u, v in G.edges()]
    colors = ['r', 'g', 'b', 'y', 'c', 'm']

    # 使用spring布局并增强社区内聚性
    pos = nx.spring_layout(G, k=4.0, iterations=800, seed=42)
    
    # 调整社区间距离
    if communities:
        # 计算每个社区的中心点
        community_centers = {}
        for i, community in enumerate(communities):
            x = sum(pos[n][0] for n in community) / len(community)
            y = sum(pos[n][1] for n in community) / len(community)
            community_centers[i] = (x, y)
        
        # 计算社区中心之间的最小距离
        min_dist = float('inf')
        centers = list(community_centers.values())
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    min_dist = dist
        
        # 调整节点位置并增加社区间距
        for i, community in enumerate(communities):
            center_x, center_y = community_centers[i]
            # 如果社区间距太小，按比例扩大
            if min_dist < 1.0:
                scale = 1.2 / min_dist
                center_x *= scale
                center_y *= scale
            
            for node in community:
                # 将节点向调整后的社区中心移动(60%)
                pos[node] = (
                    0.4 * pos[node][0] + 0.6 * center_x,
                    0.4 * pos[node][1] + 0.6 * center_y
                )

    # 两阶段绘制图形 (类似draw_group_fin.py)
    # 1. 绘制整个图的浅色背景
    nx.draw(G, pos, with_labels=False, node_size=1, width=1.8, alpha=0.1)
    
    # 添加浅色背景的边权重标签
    edge_labels = {(u, v): f"{d['weight']:.2f}" 
                  for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_color='black',
        font_size=8,
        label_pos=0.5,
        alpha=0.1  # 设置半透明效果
    )
    
    # 2. 绘制社区子图
    G_highlight = nx.Graph()
    for community in communities:
        G_highlight.update(nx.subgraph(G, community))
    
    # 绘制高亮社区
    nx.draw_networkx_nodes(G_highlight, pos,
                         node_color=[colors[i % len(colors)] 
                                   for i, comm in enumerate(communities)
                                   for _ in comm],
                         node_size=700,
                         alpha=0.9)
    
    # 绘制高亮边和权重
    edges = nx.draw_networkx_edges(G_highlight, pos, 
                                 width=[abs(G[u][v]['weight'])+1 for u,v in G_highlight.edges()],
                                 alpha=0.8, 
                                 edge_color=['red' if G[u][v]['weight'] < 0 else 'green' 
                                           for u,v in G_highlight.edges()])
    
    # 添加边权重标签
    edge_labels = {(u, v): f"{d['weight']:.2f}" 
                  for u, v, d in G_highlight.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G_highlight, pos,
        edge_labels=edge_labels,
        font_color='black',
        font_size=1,
        label_pos=0.5
    )
    
    # 绘制节点标签
    nx.draw_networkx_labels(G_highlight, pos, 
                          {n:players[n] for n in G_highlight.nodes()}, 
                          font_size=10)

    plt.title(args.function)
    plt.axis('off')
    
    # if args.output:
    #     output_path = f"/mnt/data/hqdeng7/CSCIENCE/interaction_nlp_harsanyi/group/louvain_img_1.8_1.8_community/goldfish/{args.output}"
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.show()
    
    if args.output:
        output_dir = f"/mnt/data/hqdeng7/CSCIENCE/interaction_nlp_harsanyi/group/louvain_1.3_-_weight/"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{args.output}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
