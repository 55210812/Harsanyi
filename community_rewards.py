import argparse
import re
import numpy as np
import torch
from typing import List, Dict
from interaction.harsanyi.calculate import InteractionNLP, log_rewards
from models.nlp import Calculator
from interaction.harsanyi.set_utils import flatten
from interaction.harsanyi.mask_utils import get_mask_input_function_nlp
from interaction.harsanyi.baseline_value import get_baseline_id_nlp
from utils.global_const import BASELINE_FLAG_NLP

def parse_interaction_file(file_path: str):
    """Parse interaction.txt file to get players and interactions"""
    with open(file_path, 'r') as f:
        data = f.read()

    # Parse player information
    players = {}
    player_section = re.search(r'---------- Players ----------(.*?)---------- AND Interactions', data, re.DOTALL)
    if player_section:
        for line in player_section.group(1).strip().split('\n'):
            match = re.match(r'Player (\w+): (.*)', line)
            if match:
                players[match.group(1)] = match.group(2)
    
    return players

def create_community_masks(communities: List[frozenset], all_players: List[str]) -> List[np.ndarray]:
    """Create binary masks for each community where community players are 1, others 0"""
    masks = []
    for community in communities:
        mask = np.zeros(len(all_players), dtype=bool)
        for i, player in enumerate(all_players):
            if player in community:
                mask[i] = True
        masks.append(mask)
    return masks

def calculate_community_rewards(calculator: Calculator, 
                              config: Dict,
                              input_ids: torch.Tensor,
                              attention_mask: torch.Tensor,
                              player_ids: List[List[int]],
                              communities: List[frozenset],
                              save_path: str):
    """Calculate rewards for each community"""
    interaction_nlp = InteractionNLP(calculator, config)
    
    # Create player masks for communities
    all_players = [chr(i + ord('A')) for i in range(len(player_ids))]
    community_masks = create_community_masks(communities, all_players)
    
    # Calculate rewards for each community
    rewards = []
    for mask in community_masks:
        # Convert mask to player subset format
        player_subset = [player_ids[i] for i in range(len(player_ids)) if mask[i]]
        
        # Create dummy data tuple
        data_tuple = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": 0  # dummy label
        }
        
        # Calculate rewards
        interaction_nlp(data_tuple, player_subset, save_path)
        rewards.append(interaction_nlp.get_rewards())
    
    return rewards, community_masks

def main():
    parser = argparse.ArgumentParser(description='Calculate rewards for communities')
    parser.add_argument('--file_path', required=True, help='Path to interaction.txt file')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--function', default="louvain", choices=["louvain", "gw"], 
                       help='Community detection function')
    
    args = parser.parse_args()

    # Parse interaction file to get players
    players = parse_interaction_file(args.file_path)
    all_players = sorted(players.keys())
    
    # Dummy Calculator and config (needs to be properly initialized)
    calculator = Calculator()  
    config = {
        "task": "nlp-seq-cls",
        "selected_dim": "gt-log-odds-sample=1000",
        "baseline_type": "unk",
        "gt_type": "correct",
        "background_type": "ori",
        "sort_type": "order",
        "cal_batch_size": 32,
        "verbose": True
    }
    
    # Dummy input tensors (needs proper initialization)
    input_ids = torch.zeros(1, 10)  
    attention_mask = torch.ones(1, 10)
    
    # Dummy player IDs (needs proper mapping)
    player_ids = [[i] for i in range(len(all_players))]
    
    # Run community detection (same as group.py)
    from group import main as detect_communities
    communities = detect_communities(args)
    
    # Calculate community rewards
    rewards, masks = calculate_community_rewards(
        calculator, config, input_ids, attention_mask, 
        player_ids, communities, args.output
    )
    
    # Log rewards
    player_descriptions = [players[p] for p in all_players]
    log_rewards(args.output, player_ids, masks, rewards, player_descriptions)

if __name__ == "__main__":
    main()
