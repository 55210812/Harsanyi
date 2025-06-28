import os
import os.path as osp

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def makedirs(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

def plot_mean_interaction_each_order(interactions, masks, save_path):
    n_dim = masks.shape[1]

    i_orders = np.sum(masks, axis=1).astype(int)
    mean_interactions = []

    for i_order in range(1, n_dim + 1):
        indices = i_orders == i_order
        mean_interaction = interactions[indices].mean()
        mean_interactions.append(mean_interaction)

    plt.figure()
    X = list(range(1, n_dim + 1))
    plt.bar(X, mean_interactions)
    plt.xlabel("order s=|S|")
    plt.ylabel(r"mean interaction $E_{S:|S|=s}[I(S)]$")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def plot_mean_strength_each_order(interactions, masks, save_path):
    n_dim = masks.shape[1]

    i_orders = np.sum(masks, axis=1).astype(int)
    mean_strengths = []

    for i_order in range(1, n_dim + 1):
        indices = i_orders == i_order
        mean_strength = np.abs(interactions[indices]).mean()
        mean_strengths.append(mean_strength)

    plt.figure()
    X = list(range(1, n_dim + 1))
    plt.bar(X, mean_strengths)
    plt.xlabel("order s=|S|")
    plt.ylabel(r"mean strength $E_{S:|S|=s}[|I(S)|]$")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def compare_strength_original_sparsify(before, after, save_path):
    before = before[np.argsort(-np.abs(before))]
    after = after[np.argsort(-np.abs(after))]
    X = np.arange(before.shape[0])
    plt.figure()
    plt.plot(X, np.abs(before), label="before sparsify")
    plt.plot(X, np.abs(after), label="after sparsify")
    plt.hlines(y=0, xmin=0, xmax=len(X), linestyles='dotted', colors='red')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")



def compare_original_sparsify(before, after, save_path):
    before = before[np.argsort(-before)]
    after = after[np.argsort(-after)]
    X = np.arange(before.shape[0])
    plt.figure()
    plt.plot(X, before, label="before sparsify")
    plt.plot(X, after, label="after sparsify")
    plt.hlines(y=0, xmin=0, xmax=len(X), linestyles='dotted', colors='red')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def analyze_single_folder(folder):
    save_folder = osp.join(folder, "analysis")
    makedirs(save_folder)
    masks = np.load(osp.join(folder, "masks.npy"))

    Iand_s = np.load(osp.join(folder, "Iand.npy"))
    Ior_s = np.load(osp.join(folder, "Ior.npy"))

    plot_mean_interaction_each_order(Iand_s, masks, save_path=osp.join(save_folder, "mean_interaction_Iand_after.png"))
    plot_mean_interaction_each_order(Ior_s, masks, save_path=osp.join(save_folder, "mean_interaction_Ior_after.png"))
    plot_mean_strength_each_order(Iand_s, masks, save_path=osp.join(save_folder, "mean_strength_Iand_after.png"))
    plot_mean_strength_each_order(Ior_s, masks, save_path=osp.join(save_folder, "mean_strength_Ior_after.png"))



def _get_strength_mean(interactions, masks):
    n_dim = masks.shape[1]

    i_orders = np.sum(masks, axis=1).astype(int)
    strength_means = []

    for i_order in range(1, n_dim + 1):
        indices = i_orders == i_order
        strength_mean = np.abs(interactions[indices]).mean()
        strength_means.append(strength_mean)

    return np.array(strength_means)


def get_strength_mean_single(folder):
    masks = np.load(osp.join(folder, "before_sparsify", "masks.npy"))
    rewards = np.load(osp.join(folder, "before_sparsify", "rewards.npy"))
    Iand = np.load(osp.join(folder, "before_sparsify", "Iand.npy")) * 0.5
    Ior = np.load(osp.join(folder, "before_sparsify", "Ior.npy")) * 0.5

    Iand_s = np.load(osp.join(folder, "after_sparsify", "Iand.npy"))
    Ior_s = np.load(osp.join(folder, "after_sparsify", "Ior.npy"))

    before_and = _get_strength_mean(Iand, masks)
    before_or = _get_strength_mean(Ior, masks)
    after_and = _get_strength_mean(Iand_s, masks)
    after_or = _get_strength_mean(Ior_s, masks)
    return before_and, before_or, after_and, after_or


def get_strength_mean_all(folders):
    before_and_list = []
    before_or_list = []
    after_and_list = []
    after_or_list = []
    for folder in folders:
        before_and, before_or, after_and, after_or = get_strength_mean_single(folder=folder)
        before_and_list.append(before_and)
        before_or_list.append(before_or)
        after_and_list.append(after_and)
        after_or_list.append(after_or)
    return np.array(before_and_list).mean(axis=0), \
           np.array(before_or_list).mean(axis=0), \
           np.array(after_and_list).mean(axis=0), \
           np.array(after_or_list).mean(axis=0)



def compare_mean_interaction_strength(before, after, save_path):
    plt.figure()
    plt.plot(np.arange(1, len(before) + 1), before, label="before sparsify")
    plt.plot(np.arange(1, len(after) + 1), after, label="after sparsify")
    plt.xlabel("order s=|S|")
    plt.ylabel(r"mean interaction strength $E_{x\in\Omega}E_{S:|S|=s}[|I(S|x)|]$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")

def plot_mean_interaction_strength(interactions, save_path):
    plt.figure()
    plt.plot(np.arange(1, len(interactions) + 1), interactions)
    plt.xlabel("order s=|S|")
    plt.ylabel(r"mean interaction strength $E_{x\in\Omega}E_{S:|S|=s}[|I(S|x)|]$")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")
