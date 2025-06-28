import os
import json
from collections import defaultdict

def process_sentences(input_path, output_dir):
    # 读取输入文件
    with open(input_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]
    
    # 生成单词映射表
    word_to_id = {}
    current_id = 0
    player_words = {}
    
    # 处理每个句子
    for idx, sentence in enumerate(sentences):
        words = sentence.split()
        # 记录单词ID并生成映射
        for word in words:
            if word not in word_to_id:
                word_to_id[word] = current_id
                current_id += 1
        # 保存原始句子结构
        player_words[str(idx)] = words
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 写入player_words.json
    with open(os.path.join(output_dir, 'player_words.json'), 'w', encoding='utf-8') as f:
        json.dump(player_words, f, indent=4)
    
    # 写入player_ids_from_word.json
    with open(os.path.join(output_dir, 'player_ids_from_word.json'), 'w', encoding='utf-8') as f:
        json.dump(word_to_id, f, indent=4)

if __name__ == "__main__":
    input_file = "datasets/custom-imdb-for-bertweet-nips2024-ucb/sentences.txt"
    output_dir = "players/custom-imdb-for-bertweet-nips2024-ucb-czy/players-all"
    process_sentences(input_file, output_dir)