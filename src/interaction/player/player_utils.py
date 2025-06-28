import os
import random
import json
import numpy as np

from .invalid import get_invalid_words, get_invalid_ids
from utils import save_json
from typing import List


def select_player_words_random(sentences, player_num, seed=0, save_dir = None, file_name ='player_words', is_valid=True):
    """
    Rrandomly choose the words (*seperated by space*) of the player from valid words in the sentences
    Example: "Without a doubt, one of Tobe Hoppor's best! Epic storytellng, great special effects, and The Spacegirl (vamp me baby!)."
    - "doubt," is considered a word, rather than "doubt"
    - "Hoppor's" is considered a word, rather than "Hoppor"
    - "baby!)" is considered a word, rather than "baby"

    params
    =======
    sentences: list
            the list of input sentences [str1,str2,str3,...]
    player_num: int
            the num of the player words in one sentences, commomly, player_num<=10
    save_dir: str
            the save dictory to save the player words, in form of json file.

    return 
    =======
    player_words: list
            the list of the player words [[word1,word2,...],...]
    """
    invalid_words = get_invalid_words()
    player_words_all = []
    for idx, sentence in enumerate(sentences):
        if is_valid:
            player_words = [word for word in sentence.split(' ') if word.lower() not in invalid_words and word != '']  # [Important] add .lower() to avoid case sensitive
        else:
            player_words = [word for word in sentence.split(' ') if word != '']

        sample_index = np.random.RandomState(seed=idx + seed).choice(
            [i for i in range(len(player_words))], size=player_num, replace=False).tolist()

        sample_index.sort()
        player_words = [player_words[idx] for idx in sample_index]
        player_words_all.append(player_words)

    if save_dir is not None:
        save_json(player_words_all, save_dir, f'{file_name}.json')

    return player_words_all


def select_player_words_all(sentences, save_dir=None, file_name='player_words', is_valid=True):
    """
    choose all words (separated by space) in a sentence as players, instead of randomly selecting a few words
    """
    invalid_words = get_invalid_words()
    player_words_all = []
    for idx, sentence in enumerate(sentences):
        if is_valid:
            player_words = [word for word in sentence.split(' ') if
                            word.lower() not in invalid_words and word != '']  # [Important] add .lower() to avoid case sensitive
        else:
            player_words = [word for word in sentence.split(' ') if word != '']

        player_words_all.append(player_words)

    if save_dir is not None:
        save_json(player_words_all, save_dir, f'{file_name}.json')

    return player_words_all


# todo: 这里好像还是有问题。如果用的with_indent, 可能有些词就找不到了，比如 (vamp 这种带了半个括号的
def get_player_ids_from_word(tokenizer, sentences, player_words_all, save_dir=None, file_name='player_ids_from_word'):
    """
    Based on the player words, and the given sentences, get the player's token postions in sentencs

    params
    =======
    tokenizer:
            tokenizer
    sentences: list
            the list of input sentences [str1,str2,str3,...]
    player_words: int
            the list of the player words [[word1,word2,...],...]
    save_dir: str
            the save dictory to save the player words, in form of json file.

    return 
    =======
    player_words: list
            the list of the player words [[word1,word2,...],...]
    """
    assert len(sentences) == len(player_words_all)
    player_ids_all = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        player_words = player_words_all[i]
        input_ids = tokenizer.encode(sentence)
        
        player_ids = []
        pointer = 0
        for word in player_words:
            ### 20250104 rqh: with_indent is now absorbed into the tokenizer_kwargs, so no need to do it here
            # if with_indent:
            #     word = ' ' + word

            # Since we are encoding a single word, we don't add special tokens to it, in order to get the exact token ids
            word_ids = tokenizer.encode(word, add_special_tokens=False) # important to prevent adding special tokens at the beginning and end of the word
            while True:
                if pointer == len(input_ids):
                    raise RuntimeError('Did not find corresponding words')
                if input_ids[pointer:pointer + len(word_ids)] == word_ids:
                    player_ids.append([pos for pos in range(pointer, pointer + len(word_ids))])
                    pointer += len(word_ids)
                    break
                pointer += 1
        player_ids_all.append(player_ids)

    if save_dir is not None:
        save_json(player_ids_all, save_dir, f'{file_name}.json')

    return player_ids_all


def get_player_ids_from_token(tokenizer, sentences, save_dir=None, file_name="player_ids_from_token"):
    """
    Based on the tokenizer, get the player's token positions in sentences
    ** Each valid token is considered a player **

    params
    =======
    tokenizer:
            tokenizer
    sentences: list
            the list of input sentences [str1,str2,str3,...]
    save_dir: str
            the save dictory to save the player words, in form of json file.

    return 
    =======
    player_words: list
            the list of the player words [[word1,word2,...],...]
    """
    invalid_token_ids = get_invalid_ids(tokenizer)
    player_ids_all = []
    for sentence in sentences:
        input_ids = tokenizer.encode(sentence)
        player_ids = [[pos] for pos in range(len(input_ids)) if input_ids[pos] not in invalid_token_ids]
        player_ids_all.append(player_ids)

    if save_dir is not None:
        save_json(player_ids_all, save_dir, f'{file_name}.json')

    return player_ids_all


def get_player_words_from_ids(tokenizer, input_ids: List, players):
    """
    To get nlp selected player word
    :tokenizer
    :the input of sentence which tokenized
    :selected player index
    """
    descriptions = []
    for player in players:
        d = tokenizer.decode(input_ids[player])
        descriptions.append(d)
    return descriptions


def select_player_grid2d_random(grid_num_side, player_num, seed, no_border=True):
    """
    param:
        grid_num_side: int, the number of grids in one side
        grid_select_num, the number of alternative grids for selection
        player_num:int, the number of players to be selected
    return:
        the player lists
    """
    grid_num_tot = grid_num_side ** 2  # total number of grids
    if no_border: # if no_border, the outermost grids are discarded
        grid_select_num = grid_num_side - 2
    else:
        grid_select_num = grid_num_side
    random_nums = np.random.RandomState(seed).choice(np.arange(grid_select_num), player_num, replace=False).tolist()

    # sort the random numbers
    random_nums.sort()

    if no_border: # reorganize the random numbers from a 1D list of size (grid_num_side-2)^2 to a 2D list of size grid_num_side^2
        random_nums = [(int(np.floor(number / (grid_num_side-2))) + 1) * grid_num_side + 1 + number % (grid_num_side-2)
                          for number in random_nums]

    groups = [[number] for number in random_nums]
    remaining_numbers = [num for num in range(grid_num_tot) if num not in random_nums]
    return groups, remaining_numbers


if __name__ == '__main__':
    select_player_grid2d_random(8, 10, 0)

