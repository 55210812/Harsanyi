import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import doctest
from typing import Union, Iterable, List, Tuple, Callable
from tqdm import tqdm
from itertools import combinations


def generate_all_masks(length: int, sort_type: str, k: int = None) -> list:
    """
    Generate all possible masking states for a binary sequences of length 'length'
    :param length: int, the length of the binary sequence
    :param sort_type: str, the type of sorting
        - "order": sort by the order |S|
        - "binary": sort by the binary representation
    :param k: int, optional, maximum order to generate (None means all orders)
    :return masks: list of bools, the list of all possible masks
    """
    max_order = length if k is None else min(k, length)
    masks = []
    if sort_type == "order":
        #all_S = []    
        for j in range(1,max_order+1):
            for subset in combinations(range(length),j):
                mask = [False] * length
                for i in subset:
                    mask[i] = True
                masks.append(mask)
        # for order in range(max_order+1): # 0 to max_order
        #     all_S_of_order = list(combinations(np.arange(length), order))
        #     all_S.extend(all_S_of_order)
        # masks = np.zeros((2**length, length))
        # for i, S in enumerate(all_S):
        #     masks[i, S] = 1
        # masks = [[bool(int(item)) for item in mask] for mask in masks] # list of bools
    elif sort_type == "binary":
        masks = list(range(2**length))
        masks = [np.binary_repr(mask, width=length) for mask in masks]
        masks = [[bool(int(item)) for item in mask] for mask in masks]
    else:
        raise NotImplementedError(f"sort_type [{sort_type}] is not implemented.")
    return masks

def generate_all_masks_re(length: int, sort_type: str, communities: list = None) -> list:
    """
    Generate all possible masking states for a binary sequences of length 'length'
    :param length: int, the length of the binary sequence
    :param sort_type: str, the type of sorting
        - "order": sort by the order |S|
        - "binary": sort by the binary representation
    :param communities: list, optional, list of communities where each community is a list of indices
    :return masks: list of bools, the list of all possible masks
    :raises ValueError: if communities is None or empty list
    """
    if communities is None or len(communities) == 0:
        raise ValueError("communities cannot be None or empty list")
    print(length)
    masks = []
    # Generate masks based on communities
    mask = [False] * length
    masks.append(mask)
    for community in communities:
        print(f"community:{community}")
        mask = [False] * length
        for i in community:
            print(i)
            mask[i-1] = True
        masks.append(mask)
    mask = [True] * length
    masks.append(mask)
    return masks
    # # Original behavior when communities is None
    # if sort_type == "order":
    #     max_order = length
    #     for j in range(1, max_order+1):
    #         for subset in combinations(range(length), j):
    #             mask = [False] * length
    #             for i in subset:
    #                 mask[i] = True
    #             masks.append(mask)
    # elif sort_type == "binary":
    #     masks = list(range(2**length))
    #     masks = [np.binary_repr(mask, width=length) for mask in masks]
    #     masks = [[bool(int(item)) for item in mask] for mask in masks]
    # else:
    #     raise NotImplementedError(f"sort_type [{sort_type}] is not implemented.")
    #return masks

def generate_all_communities(sample_id: int) -> list:
    communities = []
    try:
        file_path = f"group/louvain_shapley_3 copy/and_interactions_s{sample_id}.png.txt"#manual #todo: 是否需要设置为系统参数
        print(file_path)
        #file_path = f"group/all-louvain_gai_3/and_interactions_s{sample_id}.png.txt"#all
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if line.startswith("社区"):
                # Extract the letter list between brackets
                letter_str = line.split('[')[1].split(']')[0]
                letters = [x.strip().strip("'") for x in letter_str.split(',')]
                
                # Convert letters to positions (A=1, B=2, etc.)
                community = [ord(c.upper()) - ord('A') + 1 for c in letters]
                communities.append(community)
                
    except FileNotFoundError:
        print(f"Community file not found for sample_id {sample_id}")
    
    # Generate all combinations from 2 to n-1 communities
    combined_communities = []
    n = len(communities)
    for k in range(2, n):  # From 2 to n-1
        for combo in combinations(communities, k):
            # Merge all communities in the combo and deduplicate
            combined = list(set(sum(combo, [])))
            combined_communities.append(combined)
    
    return communities + combined_communities



def set_to_index(A):
    '''
    convert a boolean mask to an index
    :param A: <np.ndarray> bool (n_dim,)
    :return: an index

    [In] set_to_index(np.array([1, 0, 0, 1, 0]).astype(bool))
    [Out] 18
    '''
    assert len(A.shape) == 1
    A_ = A.astype(int)
    return np.sum([A_[-i-1] * (2 ** i) for i in range(A_.shape[0])])


def is_A_subset_B(A, B):
    '''
    Judge whether $A \subseteq B$ holds
    :param A: <numpy.ndarray> bool (n_dim, )
    :param B: <numpy.ndarray> bool (n_dim, )
    :return: Bool
    '''
    assert A.shape[0] == B.shape[0]
    return np.all(np.logical_or(np.logical_not(A), B))


def is_A_subset_Bs(A, Bs):
    '''
    Judge whether $A \subseteq B$ holds for each $B$ in 'Bs'
    :param A: <numpy.ndarray> bool (n_dim, )
    :param Bs: <numpy.ndarray> bool (n, n_dim)
    :return: Bool
    '''
    assert A.shape[0] == Bs.shape[1]
    is_subset = np.all(np.logical_or(np.logical_not(A), Bs), axis=1)
    return is_subset


def select_subset(As, B):
    '''
    Select A from As that satisfies $A \subseteq B$
    :param As: <numpy.ndarray> bool (n, n_dim)
    :param B: <numpy.ndarray> bool (n_dim, )
    :return: a subset of As
    '''
    assert As.shape[1] == B.shape[0]
    is_subset = np.all(np.logical_or(np.logical_not(As), B), axis=1)
    return As[is_subset]


def set_minus(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    '''
    calculate A/B
    :param A: <numpy.ndarray> bool (n_dim, )
    :param B: <numpy.ndarray> bool (n_dim, )
    :return: A\B

    >>> set_minus(A=np.array([1, 1, 0, 0, 1, 1], dtype=bool), B=np.array([1, 0, 0, 0, 0, 1], dtype=bool))
    array([False,  True, False, False,  True, False])

    >>> set_minus(A=np.array([1, 1, 0, 0, 1, 1], dtype=bool), B=np.array([1, 0, 1, 0, 0, 1], dtype=bool))
    array([False,  True, False, False,  True, False])
    '''
    assert A.shape[0] == B.shape[0] and len(A.shape) == 1 and len(B.shape) == 1
    A_ = A.copy()
    A_[B] = False
    return A_


def get_subset(A):
    '''
    Generate the subset of A
    :param A: <numpy.ndarray> bool (n_dim, )
    :return: subsets of A

    >>> get_subset(np.array([1, 0, 0, 1, 0, 1], dtype=bool))
    array([[False, False, False, False, False, False],
           [False, False, False, False, False,  True],
           [False, False, False,  True, False, False],
           [False, False, False,  True, False,  True],
           [ True, False, False, False, False, False],
           [ True, False, False, False, False,  True],
           [ True, False, False,  True, False, False],
           [ True, False, False,  True, False,  True]])
    '''
    assert len(A.shape) == 1
    n_dim = A.shape[0]
    n_subsets = 2 ** A.sum()
    subsets = np.zeros(shape=(n_subsets, n_dim)).astype(bool)
    subsets[:, A] = np.array(generate_all_masks(A.sum()))
    return subsets


def flatten(x) -> List:
    '''

    Flatten an irregular list of lists

    Reference <https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists>

    [In]  flatten(((1, 2), 3, 4)) -- Note: (with many brackets) x = ( (1, 2) , 3 , 4 )
    [Out] (1, 2, 3, 4)

    :param x:
    :return:
    '''
    if isinstance(x, Iterable):
        return list([a for i in x for a in flatten(i)])
    else:
        return [x]


def generate_subset_masks(set_mask, all_masks):
    '''
    For a given S, generate its subsets L's, as well as the indices of L's in [all_masks]
    :param set_mask:
    :param all_masks:
    :return: the subset masks, the bool indice
    '''
    set_mask_ = set_mask.expand_as(all_masks)
    is_subset = torch.logical_or(set_mask_, torch.logical_not(all_masks))
    is_subset = torch.all(is_subset, dim=1)
    return all_masks[is_subset], is_subset


def generate_reverse_subset_masks(set_mask, all_masks):
    '''
    For a given S, with subsets L's, generate N\L, as well as the indices of L's in [all_masks]
    :param set_mask:
    :param all_masks:
    :return:
    '''
    set_mask_ = set_mask.expand_as(all_masks)
    is_rev_subset = torch.logical_or(set_mask_, all_masks)
    is_rev_subset = torch.all(is_rev_subset, dim=1)
    return all_masks[is_rev_subset], is_rev_subset


def generate_set_with_intersection_masks(set_mask, all_masks):
    '''
    For a given S, generate L's, s.t. L and S have intersection as well as the indices of L's in [all_masks]
    :param set_mask:
    :param all_masks:
    :return:
    '''
    set_mask_ = set_mask.expand_as(all_masks)
    have_intersection = torch.logical_and(set_mask_, all_masks)
    have_intersection = torch.any(have_intersection, dim=1)
    return all_masks[have_intersection], have_intersection


if __name__ == '__main__':
    # dim = 5
    # input = torch.randn(dim)
    # baseline = torch.FloatTensor([float(100 + 100 * i) for i in range(dim)])
    # model = nn.Linear(dim, 2)
    # calculate_all_subset_outputs_pytorch(model, input, baseline)

    # all_masks = generate_all_masks(6)
    # all_masks = torch.BoolTensor(all_masks)
    # set_mask = torch.BoolTensor([1, 0, 1, 1, 0, 0])
    # print(generate_subset_masks(set_mask, all_masks))

    # print(get_subset(np.array([1, 0, 0, 1, 0, 1]).astype(bool)))
    #
    # Bs = get_subset(np.array([1, 0, 0, 1, 0, 1]).astype(bool))
    # A = np.array([1, 0, 0, 1, 0, 0]).astype(bool)
    # print(is_A_subset_Bs(A, Bs))

    # all_masks = generate_all_masks(12)
    # all_masks = np.array(all_masks, dtype=bool)
    # set_index_list = []
    # for mask in all_masks:
    #     set_index_list.append(set_to_index(mask))
    # print(len(set_index_list), len(set(set_index_list)))
    # print(min(set_index_list), max(set_index_list))

    import doctest
    doctest.testmod()



    # S [1 0 0 1 0] subset(S) -> [4, 5]
