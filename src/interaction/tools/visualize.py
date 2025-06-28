import os
import os.path as osp

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt


# # tool function
# def generate_all_masks(length: int) -> list:
#     masks = list(range(2**length))
#     masks = [np.binary_repr(mask, width=length) for mask in masks]
#     masks = [[bool(int(item)) for item in mask] for mask in masks]
#     return masks

# sort by orders
def generate_all_masks(length: int) -> list:
    from itertools import combinations
    all_S = []
    for order in range(length+1): # 0 to length
        all_S_of_order = list(combinations(np.arange(length), order))
        all_S.extend(all_S_of_order)
    masks = np.zeros((2**length, length))
    for i, S in enumerate(all_S):
        masks[i, S] = 1
    masks = [[bool(int(item)) for item in mask] for mask in masks] # list of bools
    return masks

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

def get_Iand2reward_mat(dim):
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = []
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ================================================================================================
        # Note: v(S) = \sum_{L\subseteq S} I(S)
        mask_Ls, L_indices = generate_subset_masks(mask_S, all_masks)
        row[L_indices] = 1.
        # ================================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat

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

def get_Ior2reward_mat(dim):
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = []
    mask_empty = torch.zeros(dim).bool()
    _, empty_indice = generate_subset_masks(mask_empty, all_masks)
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ================================================================================================
        # Note: v(S) = I(\emptyset) + \sum_{L: L\union S\neq \emptyset} I(S)
        row[empty_indice] = 1.
        mask_Ls, L_indices = generate_set_with_intersection_masks(mask_S, all_masks)
        row[L_indices] = 1.
        # ================================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat


def analyse_interaction(Iand, Ior, masks, save_folder, tau=0.05):
    interaction = torch.cat([Iand, Ior], dim=0)
    sym_interaction = torch.cat([torch.relu(interaction), torch.relu(-interaction)],dim=0)

    ABS_interaction = torch.abs(interaction)
    sorted_ABS_interaction = torch.sort(ABS_interaction, axis=0, descending=True)[0]
    # 归一化
    sorted_ABS_interaction = sorted_ABS_interaction / torch.max(sorted_ABS_interaction)

    sal_interaction = torch.where(ABS_interaction >= tau * torch.max(ABS_interaction), 1.0, 0)
    sal_sym_interaction = torch.where(sym_interaction >= tau * torch.max(sym_interaction), 1.0, 0)

    sym_interaction[sym_interaction < tau * torch.max(sym_interaction)] = 0

    n_dim = np.log2(sal_interaction.shape[0]).astype(int) - 1
    masks_x2 = np.concatenate([masks, masks], axis=0)
    i_orders = np.sum(masks_x2, axis=1).astype(int)
    masks_x4 = np.concatenate([masks_x2, masks_x2], axis=0)
    i_orders_x2 = np.sum(masks_x4, axis=1).astype(int)

    salconcept_num_each_order = []
    sym_salconcept_num_each_order = []
    sym_salconcept_strength_each_order = []
    for i_order in range(1, n_dim + 1):
        indices = i_orders == i_order
        salconcept_num_each_order.append(sal_interaction[indices].sum().item())
        indices = i_orders_x2 == i_order

        sym_salconcept_num_each_order.append(
                [sal_sym_interaction[np.logical_and(indices, np.concatenate([np.ones_like(i_orders), np.zeros_like(i_orders)], axis=0))].sum().item(),
                sal_sym_interaction[np.logical_and(indices, np.concatenate([np.zeros_like(i_orders), np.ones_like(i_orders)], axis=0))].sum().item()
                ])
        sym_salconcept_strength_each_order.append(
                [sym_interaction[np.logical_and(indices, np.concatenate([np.ones_like(i_orders), np.zeros_like(i_orders)], axis=0))].sum().item(),
                sym_interaction[np.logical_and(indices,np.concatenate([np.zeros_like(i_orders), np.ones_like(i_orders)], axis=0))].sum().item()
                ])
    
    sym_salconcept_num_each_order = np.array(sym_salconcept_num_each_order)
    # plot
    FontSize = 18
    x = np.arange(2** (n_dim+1))
    fig, axes = plt.subplots(1,1,figsize=(5, 4))
    plt.plot([0, 2** (n_dim+1)], [0, 0], color='red', linestyle='--')
    ax = axes
    plt.plot(x, sorted_ABS_interaction, color='blue')
    plt.legend()
    # y=0 虚线
    ax.set_xlabel('index of concepts $S$', fontsize=FontSize)
    ax.set_ylabel(f"nomalization interaction \n  strength $|I(S|x)|$", fontsize=FontSize)
    ax.tick_params(axis='x', labelsize=FontSize-2)
    ax.tick_params(axis='y', labelsize=FontSize-2)
    plt.tight_layout()
    plt.savefig(osp.join(save_folder, 'interaction_each_order.jpg'))

    fig, axes = plt.subplots(1,1,figsize=(5, 4))
    ax = axes
    plt.bar(np.arange(1, n_dim+1), salconcept_num_each_order, color='blue', label='salient concept')
    plt.legend()
    ax.set_xlabel('order of concepts', fontsize=FontSize)
    ax.set_ylabel(f"# of salient concepts", fontsize=FontSize)
    ax.xaxis.set_ticks(np.arange(1, n_dim+1))
    ax.tick_params(axis='x', labelsize=FontSize-2)
    ax.tick_params(axis='y', labelsize=FontSize-2)
    plt.tight_layout()
    plt.savefig(osp.join(save_folder, 'salient_concept_each_order.jpg'))

    fig, axes = plt.subplots(1,1,figsize=(5, 4))
    ax = axes
    plt.bar(np.arange(1, n_dim+1), sym_salconcept_num_each_order[:, 0], color='blue', label='postive concept')
    plt.bar(np.arange(1, n_dim+1), -sym_salconcept_num_each_order[:, 1], color='red', label='negative concept')
    plt.legend()
    ax.set_xlabel('order of concepts', fontsize=FontSize)
    ax.set_ylabel(f"# of salient concepts", fontsize=FontSize)
    ax.xaxis.set_ticks(np.arange(1, n_dim+1))
    ax.tick_params(axis='x', labelsize=FontSize-2)
    ax.tick_params(axis='y', labelsize=FontSize-2)
    plt.tight_layout()
    plt.savefig(osp.join(save_folder, 'sym_salient_concept_each_order.jpg'))

def mimic_all(Iand, Ior, rewards, masks, save_folder):
    n_dim = int(np.log2(Iand.size()[0]))
    Iand2reward = get_Iand2reward_mat(n_dim)
    Ior2reward = get_Ior2reward_mat(n_dim)
    rewards_order = np.argsort(rewards)


    # TODO v(kong) 代替 I(kong)
    Iand[0] = 0
    Ior[0] = 0
    mimic_reward = rewards[0] + (Iand2reward @ Iand + Ior2reward @ Ior)

    df = pd.DataFrame()
    df['subsentence_N'] = np.arange(len(mimic_reward))
    df['Real'] = rewards[rewards_order]
    df['Fitting'] = mimic_reward[rewards_order]

    FontSize = 10
    fig, ax = plt.subplots(figsize=(3,2.5) ,nrows=1)
    # 绘制两个样本类别的点线图和散点图
    sns.lineplot(
        x="subsentence_N", y="Real", data=df, ax=ax, linestyle='-', linewidth = 1, label='Real'
    ) # color=colors[0]

    sns.scatterplot(
        x="subsentence_N", y="Fitting", data=df, ax=ax, color='cornflowerblue', s=2, label='Fitting'
    )

    ax.set_xlabel('index of concepts $S$', fontsize=FontSize)
    ax.set_ylabel("LLM's output $v(x_S)$", fontsize=FontSize)
    # ax.legend(["Real"], fontsize=FontSize-1)
    ax.legend(fontsize=FontSize-1)
    ax.tick_params(axis='x', labelsize=FontSize-2)
    ax.tick_params(axis='y', labelsize=FontSize-2)
    # 调整子图间距和整个图像的布局

    plt.tight_layout()
    plt.savefig(osp.join(save_folder, "mimic_all.pdf"), dpi=200)
    plt.close("all")
