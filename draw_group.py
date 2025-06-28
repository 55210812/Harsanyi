import networkx as nx
import matplotlib.pyplot as plt
import re
import argparse
from networkx.algorithms.community import louvain_communities,girvan_newman

def parse_interactions(file_path):
    """Parse interactions from interaction.txt"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Parse player information
    players = {}
    for line in content.split('\n'):
        if line.startswith('Player '):
            parts = line.split(':')
            if len(parts) == 2:
                node = parts[0].replace('Player ', '').strip()
                word = parts[1].strip()
                players[node] = word
    
    # Parse interaction edges
    edges = []
    print("\nParsed edge weights:")
    parse_and_interactions = True
    
    for line in content.split('\n'):
        if line.startswith('---------- OR Interactions (Pairwise Only) ----------'):
            parse_and_interactions = False
            continue
            
        if parse_and_interactions:
            match = re.match(r'I\((\w)(\w)\): ([-+]?\d*\.\d+)', line.strip())
            if match:
                node1, node2, weight = match.groups()
                weight_val = float(weight)  # 保持原始权重值
                abs_weight = abs(weight_val)  # 取绝对值，以便在绘制边时使用
                print(f"Edge {node1}-{node2}: {weight_val}")
                edges.append((node1, node2, weight_val, abs_weight))
    
    return players, edges

def detect_communities(G, function):
    """Detect communities using Louvain algorithm"""
    G_copy = G.copy()  # 创建图副本避免修改原始图
    if function == "louvain":
        return louvain_communities(G_copy, weight='abs_weight')
    if function == "gw":
        return girvan_newman(G_copy, weight='abs_weight')

def visualize_communities(G, communities, players, output_path=None,edges=None):
    """Visualize the graph with community coloring"""
    pos = nx.spring_layout(G)
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    
    plt.figure(figsize=(12, 8))
    
    # Draw nodes with community colors
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=community,
                             node_color=colors[i % len(colors)],
                             node_size=800,
                             alpha=0.8)
    
    # Debug print all edge weights before drawing
    
    # Draw edges with weight-based styling
    edge_widths = [abs(weight) * 2.0 + 1.0 for u, v, weight, _ in edges]
    edge_colors = ['red' if weight < 0 else 'green' for u, v, weight, _ in edges]
    
    nx.draw_networkx_edges(G, pos,
                         width=edge_widths,
                         alpha=0.7,
                         edge_color=edge_colors)
    
    # Only show significant edge labels (|weight| > 0.5)
    edge_labels = {
        (u, v): f"{weight:.3f}"
        for u, v, weight,_ in edges
        if abs(weight) > 0.5
    }
    nx.draw_networkx_edge_labels(G, pos,
                               edge_labels=edge_labels,
                               font_size=8,
                               font_color='blue')
    nx.draw_networkx_labels(G, pos, {n:players[n] for n in G.nodes()}, font_size=10)
    
    plt.title("Louvain Community Detection")
    plt.axis('off')
    
    if output_path:
        output_path = f"/mnt/data/hqdeng7/CSCIENCE/interaction_nlp_harsanyi/group/louvain/{args.output}"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Community Detection using Louvain Algorithm')
    parser.add_argument('--file_path', required=True, help='Path to interaction.txt file')
    parser.add_argument('--output', help='Output PNG file name')
    parser.add_argument('--function', default="louvain", choices=["louvain", "gw"], help='Community detection function (default: louvain)')
    args = parser.parse_args()

    try:
        # Parse input data
        players, edges = parse_interactions(args.file_path)
        
        # Create graph
        G = nx.Graph()
        G.add_nodes_from(players.keys())
        print("\nGraph edge weights:")
        for node1, node2, weight, _ in edges:
            G.add_edge(node1, node2, weight=weight)
            # 立即验证添加的边权重
            #print(f"Verified edge {node1}-{node2}: {G[node1][node2]['weight']} (should be {weight})")
            print(f"Graph edge {node1}-{node2}: {G[node1][node2]['weight']}")
        # Detect communities
        communities = detect_communities(G,args.function)
        for node1, node2, weight, _ in edges:
            #G.add_edge(node1, node2, weight=weight)
            print(f"Verified edge {node1}-{node2}: {G[node1][node2]['weight']} (should be {weight})")
        # Print community results
        print("Louvain Community Detection Results:")
        for i, community in enumerate(communities):
            print(f"Community {i+1}: {[players[n] for n in community]}")
        
        # Visualize
        visualize_communities(G, communities, players, args.output,edges)
    
    except Exception as e:
        print(f"Error: {e}")
        exit(1)