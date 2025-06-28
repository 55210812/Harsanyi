import argparse
import re
import networkx as nx
from networkx.algorithms.community import louvain_communities
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import os

def load_communities(interaction_file, method="louvain"):
    """Load community structure from interaction file"""
    with open(interaction_file, 'r') as f:
        data = f.read()

    # Parse players
    players = {}
    player_section = re.search(r'---------- Players ----------(.*?)---------- AND Interactions', data, re.DOTALL)
    if player_section:
        for line in player_section.group(1).strip().split('\n'):
            match = re.match(r'Player (\w+): (.*)', line)
            if match:
                players[match.group(1)] = match.group(2)

    # Parse AND Interactions
    G = nx.Graph()
    interaction_section = re.search(r'---------- AND Interactions \(Pairwise Only\) ----------(.*?)(?:Sum: |\Z)', data, re.DOTALL)
    if interaction_section:
        for line in interaction_section.group(0).split('\n'):
            match = re.match(r'I\((\w+)\): ([+-]?\d*\.\d+)', line)
            if match:
                nodes = list(match.group(1))
                weight = float(match.group(2))
                abs_weight = abs(weight)
                G.add_edge(nodes[0], nodes[1], weight=weight, abs_weight=abs_weight)

    # Check if graph has edges with non-zero weights
    if len(G.edges()) == 0:
        raise ValueError("Graph has no edges - check your interaction file format")
    
    total_weight = sum(data['abs_weight'] for u, v, data in G.edges(data=True))
    if total_weight == 0:
        raise ValueError("All edge weights are zero - cannot perform community detection")

    # Detect communities
    if method == "louvain":
        try:
            communities = list(louvain_communities(G, weight='abs_weight'))
        except Exception as e:
            raise ValueError(f"Community detection failed: {str(e)}. Check your input data.")
    else:
        raise ValueError(f"Unsupported method: {method}")

    return G, [frozenset(community) for community in communities], players

def visualize_dendrogram(communities):
    """Visualize community structure as dendrogram"""
    # Create linkage matrix
    player_ids = [i for com in communities for i in com]
    community_labels = [j for j, com in enumerate(communities) for _ in com]
    
    # Create condensed distance matrix
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = np.zeros((len(player_ids), len(player_ids)))
    for i in range(len(player_ids)):
        for j in range(len(player_ids)):
            if community_labels[i] != community_labels[j]:
                dist_matrix[i,j] = 1
    condensed_dist = pdist(dist_matrix)
    
    # Perform hierarchical clustering
    Z = linkage(condensed_dist, 'single')
    
    # Plot dendrogram
    plt.figure(figsize=(10, 5))
    plt.rcParams['lines.linewidth'] = 5
    dendrogram(Z, labels=player_ids)
    plt.title("Community Dendrogram")
    plt.ylabel("Distance")
    plt.show()

def extract_player_letters(interaction_file):
    """Extract last letters from player names in interaction file"""
    with open(interaction_file, 'r') as f:
        content = f.read()
    
    # Find all player definitions
    player_matches = re.finditer(r'Player (\w+):', content)
    letters = []
    for match in player_matches:
        player_name = match.group(1)
        letters.append(player_name[-1].upper())  # Get last letter
    
    return letters

def get_combination_index(players, results_dir, sample_num=0):
    """Calculate index in rewards array using last letters of players"""
    # Get interaction_sum.txt path
    interaction_sum_path = f"{results_dir}/sample{sample_num}/interaction_sum.txt"
    
    # Get ordered list of player letters
    with open(interaction_sum_path, 'r') as f:
        content = f.read()
    
    # Extract player letters from interaction_sum.txt
    player_matches = re.finditer(r'Player (\w+):', content)
    player_letters = []
    for match in player_matches:
        player_name = match.group(1)
        player_letters.append(player_name[-1].upper())
    
    letter_to_idx = {letter: idx for idx, letter in enumerate(player_letters)}
    
    # If there is only one player, return the index of their last letter
    if len(players) == 1:
        return letter_to_idx[players[0].upper()]
    
    # For combinations, we need to match the exact order used in rewards.npy
    # First get all combinations from interaction_sum.txt
    and_interactions = re.findall(r'I\(([A-Za-z]+)\):', content)
    
    # Create mapping of sorted combination to index
    combo_to_index = {}
    for i, combo in enumerate(and_interactions):
        if len(combo) > 1:  # Only multi-player combinations
            # Get last letters and sort them
            last_letters = [p[-1].upper() for p in combo]
            sorted_combo = ''.join(sorted(last_letters))
            combo_to_index[sorted_combo] = 26 + i  # After single players
    
    # Look up our combination
    combo_letters = [p[-1].upper() for p in players]
    sorted_combo = ''.join(sorted(combo_letters))
    
    if sorted_combo in combo_to_index:
        return combo_to_index[sorted_combo]
    
    raise ValueError(f"Combination {sorted_combo} not found in rewards data")

def get_rewards_for_community(community, sample_num=0, results_dir=None):
    """Get combined interaction value for a community from rewards.npy"""
    if results_dir is None:
        results_dir = "/mnt/data/hqdeng7/CSCIENCE/interaction_nlp_harsanyi/results/20250318_harsanyi_sum_optimization_0.025/result/dataset=custom-imdb-for-bertweet-nips2024-ucb-test_model=BERTweet#pretrain_seed=0/players=players-manual_lbl=correct_baseline=unk_bg=ori#/data"
    
    sample_path = f"{results_dir}/sample{sample_num}/rewards.npy"
    
    if not os.path.exists(sample_path):
        print(f"Warning: {sample_path} not found")
        return None
    
    rewards = np.load(sample_path)
    
    try:
        # Get index for the full community combination
        combo_index = get_combination_index(community, results_dir, sample_num)
        return rewards[combo_index]
    except IndexError:
        print(f"Warning: No reward data for combination {community}")
        return None
    except Exception as e:
        print(f"Error processing rewards: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Visualize community structure and analyze rewards')
    parser.add_argument('--interaction_file', required=True, help='Path to interaction.txt file')
    parser.add_argument('--results_dir', help='Base path to results directory containing rewards.npy')
    parser.add_argument('--sample_num', type=int, default=0, help='Sample number to analyze (default: 0)')
    parser.add_argument('--output', help='Output file name for dendrogram plot')
    
    args = parser.parse_args()

    # Load communities
    G, communities, players = load_communities(args.interaction_file)
    
    # Visualize
    visualize_dendrogram(communities)
    
    # Process rewards for each community
    txt_path = f"/mnt/data/hqdeng7/CSCIENCE/interaction_nlp_harsanyi/group/tree/{args.output}.txt"
    with open(txt_path, 'w') as f:
        for i, community in enumerate(communities):
            rewards = get_rewards_for_community(
                community,
                sample_num=args.sample_num,
                results_dir=args.results_dir
            )
            print(f"Community {i} (size: {len(community)}):")
            print(f"Players: {[players[n] for n in community]}")
            print(f"Rewards: {rewards}\n")
            f.write(f"Community {i} (size: {len(community)}):\n")
            f.write(f"Players: {[players[n] for n in community]}\n")
            f.write(f"Rewards: {rewards}\n\n")
    # Save plot and results if output specified
    if args.output:
        output_path = f"/mnt/data/hqdeng7/CSCIENCE/interaction_nlp_harsanyi/group/tree/{args.output}.jpg"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
