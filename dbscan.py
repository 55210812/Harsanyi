import argparse
import networkx as nx
import re
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.cluster import DBSCAN

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Community detection from interaction data using DBSCAN clustering')
    parser.add_argument('--file_path', required=True, help='Path to interaction.txt file')
    parser.add_argument('--output', help='Output PNG file name')
    parser.add_argument('--eps', type=float, default=0.1, help='DBSCAN eps parameter (default: 0.1)')
    parser.add_argument('--min_samples', type=int, default=6, help='DBSCAN min_samples parameter (default: 5)')
    
    args = parser.parse_args()

    # 读取文件
    with open(args.file_path, 'r') as f:
        data = f.read()

    # 解析玩家信息
    players = {}
    # player_section = re.search(r'---------- Players ----------(.*?)---------- AND Interactions \(Pairwise Only\)', data, re.DOTALL)
    # if player_section:
    #     for line in player_section.group(1).strip().split('\n'):
    #         match = re.match(r'Player (\w+): (.*)', line)
    #         if match:
    #             players[match.group(1)] = match.group(2)
    player_section = re.search(r'Player \d+: block_\d+_\d+', data)
    if player_section:
        #print("Player section found")
        for line in data.split('\n'):
            if line.startswith('Player'):
                match = re.match(r'Player (\d+): (block_\d+_\d+)', line)
                if match:
                    players[match.group(1)] = match.group(2)
    # 解析AND Interactions并构建图
    G = nx.Graph()
    # interaction_section = re.search(r'---------- AND Interactions.*?---------- OR Interactions', data, re.DOTALL)
    # if interaction_section:
    #     for line in interaction_section.group(0).split('\n'):
    #         match = re.match(r'I\((\w+)\): ([+-]?\d*\.\d+)', line)
    #         if match:
    #             nodes = list(match.group(1))
    #             weight = float(match.group(2))
    #             abs_weight = abs(weight)
    #             G.add_edge(nodes[0], nodes[1], weight=weight, abs_weight=abs_weight)
    # interaction_section = re.search(r'---------- AND Interactions \(Pairwise Only\) ----------(.*?)(?:Sum: |\Z)', data, re.DOTALL)
    # if interaction_section:
    #     seen_edges = set()
    #     for line in interaction_section.group(1).strip().split('\n'):
    #         match = re.match(r'I\((\w+)\): ([+-]?\d*\.\d+)', line)
    #         if match:
    #             edge = match.group(1)
    #             if edge not in seen_edges:  # 跳过重复边
    #                 seen_edges.add(edge)
    #                 nodes = list(edge)
    #                 weight = float(match.group(2))
    #                 abs_weight = abs(weight)
    #                 G.add_edge(nodes[0], nodes[1], weight=weight, abs_weight=abs_weight)
    for line in data.split('\n'):
        if line.startswith('I('):
            #print("Interaction section found")
            match = re.match(r'I\((\d+),(\d+)\): ([+-]?\d*\.\d+)\s+\(\[block_\d+_\d+\]\[block_\d+_\d+\]\)', line)
            if match:
                #print(f"Match found: {match.group(0)}")
                node1, node2 = match.group(1), match.group(2)
                weight = float(match.group(3))
                #print(f"Adding edge: {node1}, {node2}, weight: {weight}")
                abs_weight = abs(weight)
                G.add_edge(node1, node2, weight=weight, abs_weight=abs_weight)


    # 构建特征矩阵用于DBSCAN聚类
    nodes = list(G.nodes())
    n = len(nodes)
    
    if n == 0:
        print("Warning: No nodes found, returning empty communities")
        communities = []
    else:
        # 创建特征矩阵(使用节点度数和聚类系数作为特征)
        features = []
        for node in nodes:
            degree = G.degree(node)
            clustering = nx.clustering(G, node)
            features.append([degree, clustering])
        
        features = np.array(features)
        
        # 执行DBSCAN聚类
        dbscan = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit(features)
        cluster_labels = dbscan.labels_
        
        # 将聚类结果转换为社区格式
        communities = []
        unique_labels = set(cluster_labels)
        
        # 处理噪声点(标签为-1)
        if -1 in unique_labels:
            print(f"DBSCAN found {sum(cluster_labels == -1)} noise points")
            unique_labels.remove(-1)
        
        for label in unique_labels:
            community = [nodes[i] for i in range(n) if cluster_labels[i] == label]
            if community:  # 只添加非空社区
                communities.append(community)

    # 打印社区结果
    output_path = f"/mnt/data/hqdeng7/CSCIENCE/interaction_nlp_harsanyi/group/dbscan_img/dog/{args.output}.txt"
    print("DBSCAN聚类社区检测结果:")
    with open(output_path, 'w') as f:
        f.write("DBSCAN聚类社区检测结果:\n")
        for i, community in enumerate(communities):
            print(f"社区 {i+1}: {[players[n] for n in community]}")
            f.write(f"社区 {i+1}: {[n for n in community]}: {[players[n] for n in community]}\n")

    # 可视化(保持与group.py相同的可视化代码)
    # weights = [abs(G[u][v]['weight'])*2 for u,v in G.edges()]
    # edge_colors = ['red' if G[u][v]['weight'] < 0 else 'green' for u, v in G.edges()]
    # colors = ['r', 'g', 'b', 'y', 'c', 'm']

    # pos = nx.spring_layout(G, k=2.0, iterations=800, seed=42)
    
    # if communities:
    #     community_centers = {}
    #     for i, community in enumerate(communities):
    #         x = sum(pos[n][0] for n in community) / len(community)
    #         y = sum(pos[n][1] for n in community) / len(community)
    #         community_centers[i] = (x, y)
        
    #     min_dist = float('inf')
    #     centers = list(community_centers.values())
    #     for i in range(len(centers)):
    #         for j in range(i+1, len(centers)):
    #             dx = centers[i][0] - centers[j][0]
    #             dy = centers[i][1] - centers[j][1]
    #             dist = math.sqrt(dx*dx + dy*dy)
    #             if dist < min_dist:
    #                 min_dist = dist
        
    #     for i, community in enumerate(communities):
    #         center_x, center_y = community_centers[i]
    #         if min_dist < 1.0:
    #             scale = 1.2 / min_dist
    #             center_x *= scale
    #             center_y *= scale
            
    #         for node in community:
    #             pos[node] = (
    #                 0.4 * pos[node][0] + 0.6 * center_x,
    #                 0.4 * pos[node][1] + 0.6 * center_y
    #             )

    # # 两阶段绘制图形
    # nx.draw(G, pos, with_labels=False, node_size=1, width=0.7, alpha=0.1)
    
    # edge_labels = {(u, v): f"{d['weight']:.2f}" 
    #               for u, v, d in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(
    #     G, pos,
    #     edge_labels=edge_labels,
    #     font_color='black',
    #     font_size=8,
    #     label_pos=0.5,
    #     alpha=0.1
    # )
    
    # G_highlight = nx.Graph()
    # for community in communities:
    #     G_highlight.update(nx.subgraph(G, community))
    
    # nx.draw_networkx_nodes(G_highlight, pos,
    #                      node_color=[colors[i % len(colors)] 
    #                                for i, comm in enumerate(communities)
    #                                for _ in comm],
    #                      node_size=700,
    #                      alpha=0.9)
    
    # edges = nx.draw_networkx_edges(G_highlight, pos, 
    #                              width=[abs(G[u][v]['weight'])*2+1 for u,v in G_highlight.edges()],
    #                              alpha=0.7, 
    #                              edge_color=['red' if G[u][v]['weight'] < 0 else 'green' 
    #                                        for u,v in G_highlight.edges()])
    
    # edge_labels = {(u, v): f"{d['weight']:.2f}" 
    #               for u, v, d in G_highlight.edges(data=True)}
    # nx.draw_networkx_edge_labels(
    #     G_highlight, pos,
    #     edge_labels=edge_labels,
    #     font_color='black',
    #     font_size=8,
    #     label_pos=0.5
    # )
    
    # nx.draw_networkx_labels(G_highlight, pos, 
    #                       {n:players[n] for n in G_highlight.nodes()}, 
    #                       font_size=10)

    # plt.title("DBSCAN Clustering")
    # plt.axis('off')
    
    # if args.output:
    #     output_path = f"/mnt/data/hqdeng7/CSCIENCE/interaction_nlp_harsanyi/group/dbscan_shapley_2/{args.output}"
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    main()