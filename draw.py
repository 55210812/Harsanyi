import networkx as nx
import matplotlib.pyplot as plt
import re
import argparse

def parse_interactions(file_path):
    """Parse AND Interactions from interaction.txt"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract AND Interactions section
    and_section = re.search(r'---------- AND Interactions \(Pairwise Only\) ----------(.*?)---------- OR Interactions', 
                           content, re.DOTALL)
    if not and_section:
        raise ValueError("AND Interactions section not found")
    
    # Parse each interaction line
    edges = []
    for line in and_section.group(1).split('\n'):
        match = re.match(r'I\((\w)(\w)\): ([-+]?\d*\.\d+)\s*\((.*?)\)', line.strip())
        if match:
            node1, node2, weight, words = match.groups()
            edges.append((node1, node2, float(weight), words))
    
    return edges

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Draw AND Interactions Graph')
    parser.add_argument('--file_path', required=True, help='Path to interaction.txt file')
    parser.add_argument('--output', help='Output PNG file name')
    args = parser.parse_args()

    # Create graph
    G = nx.Graph()

    # Add nodes with their corresponding words
    edges = parse_interactions(args.file_path)
    
    # Extract node-word mapping from players section
    with open(args.file_path, 'r') as f:
        lines = f.readlines()
    
    node_words = {}
    for line in lines:
        if line.startswith('Player '):
            parts = line.split(':')
            if len(parts) == 2:
                node = parts[0].replace('Player ', '').strip()
                word = parts[1].strip()
                node_words[node] = word
    
    G.add_nodes_from(node_words.keys())

    # Add edges with weights
    for node1, node2, weight, _ in edges:
        G.add_edge(node1, node2, weight=weight)

    # Draw graph with better node distribution
    pos = nx.spring_layout(G, k=2.0, iterations=100)  # Increase spacing between nodes
    weights = [abs(G[u][v]['weight'])*2 for u,v in G.edges()]  # Scale weights for visibility

    # Create node labels with single word
    node_labels = {node: f"{node}\n{word}" 
                  for node, word in node_words.items()}
    
    # Draw with larger nodes and adjusted font sizes
    nx.draw(G, pos, labels=node_labels, node_size=1500, node_color='lightblue',
            width=weights, edge_color='gray', font_size=10, font_weight='bold')

    # Add edge labels with formatted weights
    edge_labels = {(u, v): f"{data['weight']:.2f}" 
                  for u, v, data in G.edges(data=True)}
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("AND Interactions Graph")
    
    if args.output:
        output_path = f"//mnt/data/hqdeng7/CSCIENCE/interaction_nlp_harsanyi/interactions/,shapley_interaction/{args.output}"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to {output_path}")
    plt.show()
