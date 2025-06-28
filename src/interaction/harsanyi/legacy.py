# This file contains the legacy code for Harsanyi transformation.
# determine all import packages
import torch
import numpy as np
from tqdm import tqdm
from .and_or_harsanyi_utils import get_Iand2reward_mat, get_Ior2reward_mat, get_reward2Iand_mat, get_reward2Ior_mat
from typing import Callable, Union, List, Dict, Tuple
from .set_utils import generate_all_masks, flatten
import torch.nn as nn
import torch.optim as optim
import os.path as osp


def remove_noisy_and_or_harsanyi(
        I_and: torch.Tensor,
        I_or: torch.Tensor,
        threshold: float  # AOG 中的 interaction 强度占全部 interaction 的强度
) -> Tuple[torch.Tensor, torch.Tensor, List, List]:
    """

    :param I_and:
    :param I_or:
    :param threshold: the threshold for total strength of removed interactions
    :return:
    """
    if threshold == 0:
        I_and_retained_indices = list(np.arange(I_and.shape[0]).tolist())
        I_or_retained_indices = list(np.arange(I_or.shape[0]).tolist())
        return I_and, I_or, I_and_retained_indices, I_or_retained_indices

    interactions = torch.cat([I_and, I_or]).clone()
    total_strength = torch.abs(interactions).sum() + 1e-7
    strength_order = torch.argsort(torch.abs(interactions))

    removed_ratio = torch.cumsum(torch.abs(interactions[strength_order]), dim=0) / total_strength
    first_retain_id = (removed_ratio > threshold).nonzero()[0, 0]
    removed_indices = strength_order[:first_retain_id]
    retained_indices = strength_order[first_retain_id:]

    interactions[removed_indices] = 0  # set the interaction of removed patterns to zero

    I_and_retained_indices = retained_indices[retained_indices < I_and.shape[0]].tolist()
    I_or_retained_indices = retained_indices[retained_indices >= I_and.shape[0]] - I_and.shape[0]
    I_or_retained_indices = I_or_retained_indices.tolist()

    I_and_ = interactions[:I_and.shape[0]]
    I_or_ = interactions[I_and.shape[0]:]

    # return I_and_, I_or_, first_retain_id
    return I_and_, I_or_, I_and_retained_indices, I_or_retained_indices


def generate_and_or_harsanyi_remove_order_min_unfaith(
        I_and: torch.Tensor,
        I_or: torch.Tensor,
        n_greedy: int = 40  # 每次考虑 n_greedy 个 interaction pattern 作为 candidate，从中删去一个
) -> np.ndarray:
    device = I_and.device
    n_players = int(np.log2(I_and.shape[0]))
    Iand2reward = get_Iand2reward_mat(n_players).to(device)
    Ior2reward = get_Ior2reward_mat(n_players).to(device)

    rewards = Iand2reward @ I_and + Ior2reward @ I_or

    interactions = torch.cat([I_and, I_or]).clone()
    strength_order = torch.argsort(torch.abs(interactions)).tolist()  # from low-strength to high-strength
    remove_order = []

    for n_remove in tqdm(range(interactions.shape[0]), desc="calc remove order", ncols=100):
        candidates = strength_order[:n_greedy]
        to_remove = candidates[0]
        interactions_ = interactions.clone()
        interactions_[to_remove] = 0.
        I_and_ = interactions_[:I_and.shape[0]]
        I_or_ = interactions_[I_and.shape[0]:]
        rewards_aog_ = Iand2reward @ I_and_ + Ior2reward @ I_or_
        unfaith = torch.sum(torch.square(rewards - rewards_aog_))

        for i in range(1, len(candidates)):
            candidate = candidates[i]
            interactions_ = interactions.clone()
            interactions_[candidate] = 0.
            I_and_ = interactions_[:I_and.shape[0]]
            I_or_ = interactions_[I_and.shape[0]:]
            rewards_aog_ = Iand2reward @ I_and_ + Ior2reward @ I_or_
            unfaith_ = torch.sum(torch.square(rewards - rewards_aog_))
            if unfaith_ < unfaith:
                to_remove = candidate
                unfaith = unfaith_

        interactions[to_remove] = 0.
        remove_order.append(to_remove)
        strength_order.remove(to_remove)

    return np.array(remove_order)


def remove_noisy_and_or_harsanyi_given_remove_order(
        I_and: torch.Tensor,
        I_or: torch.Tensor,
        remove_order: np.ndarray,
        threshold: float = None,  # 删去的 interaction 强度占全部 interaction 的强度
        retain_num: int = None,  # 最终留下来的 interaction pattern 数量
) -> Tuple[torch.Tensor, torch.Tensor, List, List]:
    if threshold is not None:
        return _remove_noisy_given_order_remove_ratio(I_and=I_and, I_or=I_or,
                                                      remove_order=remove_order, threshold=threshold)
    if retain_num is not None:
        return _remove_noisy_given_order_retain_num(I_and=I_and, I_or=I_or,
                                                    remove_order=remove_order, retain_num=retain_num)


def _remove_noisy_given_order_remove_ratio(
        I_and: torch.Tensor,
        I_or: torch.Tensor,
        remove_order: np.ndarray,
        threshold: float,  # 删去的 interaction 强度占全部 interaction 的强度
) -> Tuple[torch.Tensor, torch.Tensor, List, List]:
    assert threshold < 1

    if threshold == 0:
        I_and_retained_indices = list(np.arange(I_and.shape[0]).tolist())
        I_or_retained_indices = list(np.arange(I_or.shape[0]).tolist())
        return I_and, I_or, I_and_retained_indices, I_or_retained_indices

    interactions = torch.cat([I_and, I_or]).clone()
    total_strength = torch.abs(interactions).sum() + 1e-7

    removed_ratio = torch.cumsum(torch.abs(interactions[remove_order]), dim=0) / total_strength
    first_retain_id = (removed_ratio > threshold).nonzero()[0, 0]
    removed_indices = remove_order[:first_retain_id]
    retained_indices = remove_order[first_retain_id:]

    interactions[removed_indices] = 0  # set the interaction of removed patterns to zero

    I_and_retained_indices = retained_indices[retained_indices < I_and.shape[0]].tolist()
    I_or_retained_indices = retained_indices[retained_indices >= I_and.shape[0]] - I_and.shape[0]
    I_or_retained_indices = I_or_retained_indices.tolist()

    I_and_ = interactions[:I_and.shape[0]]
    I_or_ = interactions[I_and.shape[0]:]

    # return I_and_, I_or_, first_retain_id
    return I_and_, I_or_, I_and_retained_indices, I_or_retained_indices


def _remove_noisy_given_order_retain_num(
        I_and: torch.Tensor,
        I_or: torch.Tensor,
        remove_order: np.ndarray,
        retain_num: int = None,  # 最终留下来的 interaction pattern 数量
) -> Tuple[torch.Tensor, torch.Tensor, List, List]:

    assert retain_num > 0

    interactions = torch.cat([I_and, I_or]).clone()
    first_retain_id = interactions.shape[0] - retain_num
    removed_indices = remove_order[:first_retain_id]
    retained_indices = remove_order[first_retain_id:]
    assert len(retained_indices) == retain_num

    interactions[removed_indices] = 0  # set the interaction of removed patterns to zero

    I_and_retained_indices = retained_indices[retained_indices < I_and.shape[0]].tolist()
    I_or_retained_indices = retained_indices[retained_indices >= I_and.shape[0]] - I_and.shape[0]
    I_or_retained_indices = I_or_retained_indices.tolist()

    I_and_ = interactions[:I_and.shape[0]]
    I_or_ = interactions[I_and.shape[0]:]

    # return I_and_, I_or_, first_retain_id
    return I_and_, I_or_, I_and_retained_indices, I_or_retained_indices


def remove_noisy_and_or_harsanyi_min_unfaith(
        I_and: torch.Tensor,
        I_or: torch.Tensor,
        threshold: float,  # 删去的 interaction 强度占全部 interaction 的强度
        n_greedy: int = 40  # 每次考虑 n_greedy 个 interaction pattern 作为 candidate，从中删去一个
) -> Tuple[torch.Tensor, torch.Tensor, List, List]:

    if threshold == 0:
        I_and_retained_indices = list(np.arange(I_and.shape[0]).tolist())
        I_or_retained_indices = list(np.arange(I_or.shape[0]).tolist())
        return I_and, I_or, I_and_retained_indices, I_or_retained_indices

    device = I_and.device
    n_players = int(np.log2(I_and.shape[0]))
    Iand2reward = get_Iand2reward_mat(n_players).to(device)
    Ior2reward = get_Ior2reward_mat(n_players).to(device)

    rewards = Iand2reward @ I_and + Ior2reward @ I_or

    interactions = torch.cat([I_and, I_or]).clone()
    interactions_original = interactions.clone()
    strength_order = torch.argsort(torch.abs(interactions)).tolist()  # from low-strength to high-strength
    removed_indices = []

    for n_remove in tqdm(range(interactions.shape[0]), desc="removing", ncols=100):
        candidates = strength_order[:n_greedy]
        to_remove = candidates[0]
        interactions_ = interactions.clone()
        interactions_[to_remove] = 0.
        I_and_ = interactions_[:I_and.shape[0]]
        I_or_ = interactions_[I_and.shape[0]:]
        rewards_aog_ = Iand2reward @ I_and_ + Ior2reward @ I_or_
        unfaith = torch.sum(torch.square(rewards - rewards_aog_))

        for i in range(1, len(candidates)):
            candidate = candidates[i]
            interactions_ = interactions.clone()
            interactions_[candidate] = 0.
            I_and_ = interactions_[:I_and.shape[0]]
            I_or_ = interactions_[I_and.shape[0]:]
            rewards_aog_ = Iand2reward @ I_and_ + Ior2reward @ I_or_
            unfaith_ = torch.sum(torch.square(rewards - rewards_aog_))
            if unfaith_ < unfaith:
                to_remove = candidate
                unfaith = unfaith_

        interactions[to_remove] = 0.
        ratio = 1 - torch.sum(torch.abs(interactions)) / (torch.sum(torch.abs(interactions_original)) + 1e-7)
        if ratio > threshold:
            interactions[to_remove] = interactions_original[to_remove]
            break
        removed_indices.append(to_remove)
        strength_order.remove(to_remove)

    I_and_ = interactions[:I_and.shape[0]]
    I_or_ = interactions[I_and.shape[0]:]
    retained_indices = np.array([i for i in range(interactions.shape[0]) if i not in removed_indices])

    I_and_retained_indices = retained_indices[retained_indices < I_and.shape[0]].tolist()
    I_or_retained_indices = retained_indices[retained_indices >= I_and.shape[0]] - I_and.shape[0]
    I_or_retained_indices = I_or_retained_indices.tolist()

    return I_and_, I_or_, I_and_retained_indices, I_or_retained_indices


def get_and_or_harsanyi_inference_func(
        all_masks: torch.Tensor,
        I_and: torch.Tensor,
        I_or: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]: # todo: 没有看懂这个函数的作用，貌似也没有其他地方的usage
    assert len(all_masks.shape) == 2

    empty_indices = torch.all(torch.logical_not(all_masks), dim=1)
    assert empty_indices.sum().item() == 1

    def inference_func(input_mask: torch.Tensor) -> torch.Tensor:
        assert all_masks.shape[1] == input_mask.shape[0]

        if torch.any(input_mask):
            act_indices_and = torch.all(torch.logical_or(torch.logical_not(all_masks), input_mask[None, :]), dim=1)
            act_indices_or = torch.any(all_masks[:, input_mask], dim=1)
            act_indices_and = torch.logical_or(empty_indices, act_indices_and)
            act_indices_or = torch.logical_or(empty_indices, act_indices_or)
        else:
            act_indices_and = empty_indices.clone()
            act_indices_or = empty_indices.clone()

        return I_and[act_indices_and].sum() + I_or[act_indices_or].sum()

    return inference_func


def reorganize_and_or_harsanyi(
        all_masks: torch.Tensor,
        I_and: torch.Tensor,
        I_or: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    To combine the single-order, zero-order components in and-or interactions together
    :param all_masks:
    :param I_and:
    :param I_or:
    :return:
    """
    I_and_ = I_and.clone()
    I_or_ = I_or.clone()

    comb_indices = torch.sum(all_masks, dim=1) <= 1

    I_and_[comb_indices] = I_and_[comb_indices] + I_or_[comb_indices]
    I_or_[comb_indices] = 0

    return I_and_, I_or_


def calculate_output_N(
        model: Union[nn.Module, Callable],
        input: torch.Tensor,
        baseline: torch.Tensor,
        all_players: Union[None, tuple] = None,
        background: Union[None, tuple] = None,
        mask_input_fn: Callable = None,
        verbose: int = 1
):
    # todo: 这些assert 需要吗？
    assert input.shape[0] == 1
    if isinstance(baseline, torch.Tensor):
        assert baseline.shape[0] == 1
    device = input.device
    if all_players is None:
        n_players = input.shape[1]
    else:
        n_players = len(all_players)
    player_masks = torch.ones(1, n_players).bool().to(device)
    _, output_N = calculate_given_subset_outputs(
        model=model, input=input, baseline=baseline,
        all_players=all_players, background=background,
        mask_input_fn=mask_input_fn, calc_bs=None,
        player_masks=player_masks, verbose=verbose
    )
    return output_N


def calculate_output_empty(
        model: Union[nn.Module, Callable],
        input: torch.Tensor,
        baseline: torch.Tensor,
        all_players: Union[None, tuple] = None,
        background: Union[None, tuple] = None,
        mask_input_fn: Callable = None,
        verbose: int = 1
):
    # todo: 这些assert 需要吗？
    assert input.shape[0] == 1
    if isinstance(baseline, torch.Tensor):
        assert baseline.shape[0] == 1
    device = input.device
    if all_players is None:
        n_players = input.shape[1]
    else:
        n_players = len(all_players)
    player_masks = torch.zeros(1, n_players).bool().to(device)
    _, output_empty = calculate_given_subset_outputs(
        model=model, input=input, baseline=baseline,
        all_players=all_players, background=background,
        mask_input_fn=mask_input_fn, calc_bs=None,
        player_masks=player_masks, verbose=verbose
    )
    return output_empty


def calculate_given_subset_outputs(
        model: Union[nn.Module, Callable],
        input: torch.Tensor,
        baseline: torch.Tensor,
        player_masks: torch.BoolTensor,
        all_players: Union[None, tuple, list] = None,
        background: Union[None, tuple, list] = None,
        mask_input_fn: Callable = None,
        calc_bs: Union[None, int] = None,
        verbose: int = 1
) -> (torch.Tensor, torch.Tensor):
    assert input.shape[0] == 1
    if isinstance(baseline, torch.Tensor):
        assert baseline.shape[0] == 1
    # ======================================================
    #     (1) First, generate the masked inputs
    # ======================================================
    if all_players is None:
        assert (background is None or len(background) == 0) and mask_input_fn is None
        masks = player_masks
        # masked_inputs = torch.where(masks, input.expand_as(masks), baseline.expand_as(masks))
    else:
        if background is None:
            background = []
        assert background is not None and mask_input_fn is not None
        all_players = np.array(all_players, dtype=object)
        grid_indices_list = []
        for i in range(player_masks.shape[0]):
            player_mask = player_masks[i].clone().cpu().numpy()
            grid_indices_list.append(list(flatten([background, all_players[player_mask]])))
            # todo: 这里要处理一下background是mask还是不mask的两种情况

        # masked_inputs = mask_input_fn(image=input, baseline=baseline, grid_indices_list=grid_indices_list)
    # # TODO
    # useful_loc = np.sort(np.array(flatten([background, [player[0] for player in all_players]])))
    # ======================================================
    #  (2) Second, calculate the rewards of these inputs
    # ======================================================
    if calc_bs is None:
        calc_bs = player_masks.shape[0]

    outputs = []
    if verbose == 1:
        pbar = tqdm(range(int(np.ceil(player_masks.shape[0] / calc_bs))), ncols=100, desc="Calc model outputs")
    else:
        pbar = range(int(np.ceil(player_masks.shape[0] / calc_bs)))
    for batch_id in pbar:
        if all_players is None:
            masks_batch = masks[batch_id * calc_bs:(batch_id + 1) * calc_bs]
            masked_inputs_batch = torch.where(masks_batch, input.expand_as(masks_batch),
                                              baseline.expand_as(masks_batch))
            # todo: 思考一下这个情况是不是也可以用mask_input_fn呢？这样可能会更通用一些，而且这样就不需要baseline一定是一个tensor了
            # todo: (continue) 可以根据不同的mask_input_fn自行调整了（比如对于nlp, baseline可以就只是个int，即一个baseline flag）
        else:
            grid_indices_batch = grid_indices_list[batch_id * calc_bs:(batch_id + 1) * calc_bs]
            masked_inputs_batch = mask_input_fn(input, baseline, grid_indices_batch)

        # masked_inputs_batch = masked_inputs_batch[:,useful_loc]
        output = model(masked_inputs_batch)
        outputs.append(output)
    outputs = torch.cat(outputs, dim=0)

    return player_masks, outputs


def calculate_all_subset_outputs(
        model: Union[nn.Module, Callable],
        input: torch.Tensor,
        baseline: torch.Tensor,
        all_players: Union[None, tuple] = None,
        background: Union[None, tuple] = None,
        mask_input_fn: Callable = None,
        calc_bs: Union[None, int] = None,
        verbose: int = 1
) -> (torch.Tensor, torch.Tensor):
    # todo: 这些assert 需要吗？
    assert input.shape[0] == 1
    if isinstance(baseline, torch.Tensor):
        assert baseline.shape[0] == 1
    device = input.device
    if all_players is None:
        n_players = input.shape[1]
    else:
        n_players = len(all_players)
    player_masks = torch.BoolTensor(generate_all_masks(n_players)).to(device)
    return calculate_given_subset_outputs(
        model=model, input=input, baseline=baseline,
        player_masks=player_masks, all_players=all_players,
        background=background, mask_input_fn=mask_input_fn,
        calc_bs=calc_bs, verbose=verbose
    )


def _train_p(
        rewards: torch.Tensor,
        loss_type: str,
        lr: float,
        niter: int,
        reward2Iand: torch.Tensor = None,
        reward2Ior: torch.Tensor = None
) -> Tuple[torch.Tensor, Dict, Dict]:
    device = rewards.device
    n_dim = int(np.log2(rewards.numel()))
    if reward2Iand is None:
        reward2Iand = get_reward2Iand_mat(n_dim).to(device)
    if reward2Ior is None:
        reward2Ior = get_reward2Ior_mat(n_dim).to(device)

    p = torch.zeros_like(rewards).requires_grad_(True)
    optimizer = optim.SGD([p], lr=0.0, momentum=0.9)

    log_lr = np.log10(lr)
    eta_list = np.logspace(log_lr, log_lr - 1, niter)

    if loss_type == "l1":
        losses = {"loss": []}
    elif loss_type.startswith("l1_on"):
        ratio = float(loss_type.split("_")[-1])
        Iand_p = torch.matmul(reward2Iand, 0.5 * rewards + p)
        Ior_p = torch.matmul(reward2Ior, 0.5 * rewards - p)
        num_noisy_pattern = int(ratio * (Iand_p.shape[0] + Ior_p.shape[0]))
        print("# noisy patterns", num_noisy_pattern)
        noisy_pattern_indices = torch.argsort(torch.abs(torch.cat([Iand_p, Ior_p]))).tolist()[:num_noisy_pattern]
        losses = {"loss": [], "noise_ratio": []}
    else:
        raise NotImplementedError(f"Loss type {loss_type} unrecognized.")

    progresses = {"I_and": [], "I_or": []}
    ckpt_id_list = generate_ckpt_id_list(niter=niter, nckpt=20)

    pbar = tqdm(range(niter), desc="Optimizing p", ncols=100)
    for it in pbar:
        Iand_p = torch.matmul(reward2Iand, 0.5 * rewards + p)
        Ior_p = torch.matmul(reward2Ior, 0.5 * rewards - p)

        if loss_type == "l1":
            loss = torch.sum(torch.abs(Iand_p)) + torch.sum(torch.abs(Ior_p))  # 02-27: L1 penalty.
            losses["loss"].append(loss.item())
        elif loss_type.startswith("l1_on"):
            loss = l1_on_given_dim(torch.cat([Iand_p, Ior_p]), indices=noisy_pattern_indices)
            losses["loss"].append(loss.item())
            losses["noise_ratio"].append(loss.item() / torch.sum(torch.abs(torch.cat([Iand_p, Ior_p]))).item())
        else:
            raise NotImplementedError(f"Loss type {loss_type} unrecognized.")

        if it + 1 < niter:
            optimizer.zero_grad()
            optimizer.param_groups[0]["lr"] = eta_list[it]
            loss.backward()
            optimizer.step()

        if it in ckpt_id_list:
            progresses["I_and"].append(Iand_p.detach().cpu().numpy())
            progresses["I_or"].append(Ior_p.detach().cpu().numpy())
            pbar.set_postfix_str(f"loss={loss.item():.4f}")

    return p.detach(), losses, progresses


def _train_p_q(
        rewards: torch.Tensor,
        loss_type: str,
        lr: float,
        momentum: float,
        niter: int,
        qbound: Union[float, torch.Tensor],
        q_tricks: bool,
        reward2Iand: torch.Tensor = None,
        reward2Ior: torch.Tensor = None,
        pq_path = None # Todo：这里要改掉
):
    device = rewards.device
    n_dim = int(np.log2(rewards.numel()))
    if reward2Iand is None:
        reward2Iand = get_reward2Iand_mat(n_dim).to(device)
    if reward2Ior is None:
        reward2Ior = get_reward2Ior_mat(n_dim).to(device)

    # 使用变化的学习率
    # log_lr = np.log10(lr)
    # eta_list = np.logspace(log_lr, log_lr - 1, niter)
    q_10dim = None
    p_10dim = None
    and_10dim = None
    or_10dim = None


    if pq_path == None:
        p = torch.zeros_like(rewards).requires_grad_(True)
        q = torch.zeros_like(rewards).requires_grad_(True)
    else: # todo: 这里还要改一下，改成仅仅是用argument里的p和q去做初始化，但是不一定是从某个path load进来的
        p = torch.tensor(np.load(osp.join(pq_path, "p.npy"))).to(device).requires_grad_(True)
        q = torch.tensor(np.load(osp.join(pq_path, "q.npy"))).to(device).requires_grad_(True)

    q_indices = torch.randperm(q.size(0))

    optimizer = optim.SGD([p, q], lr=lr, momentum=momentum)

    if loss_type == "l1":
        losses = {"loss": [], "q_norm_L1":[], "q_mean":[]}
    elif loss_type.startswith("l1_on"):
        ratio = float(loss_type.split("_")[-1])
        Iand_p = torch.matmul(reward2Iand, 0.5 * rewards + p)
        Ior_p = torch.matmul(reward2Ior, 0.5 * rewards - p)
        num_noisy_pattern = int(ratio * (Iand_p.shape[0] + Ior_p.shape[0]))
        print("# noisy patterns", num_noisy_pattern)
        noisy_pattern_indices = torch.argsort(torch.abs(torch.cat([Iand_p, Ior_p]))).tolist()[:num_noisy_pattern]
        losses = {"loss": [], "noise_ratio": [], "q_norm_L1":[], "q_mean":[]}
    else:
        raise NotImplementedError(f"Loss type {loss_type} unrecognized.")

    progresses = {"I_and": [], "I_or": []}
    ckpt_id_list = generate_ckpt_id_list(niter=niter, nckpt=20)

    pbar = tqdm(range(niter), desc="Optimizing pq", ncols=100)
    for it in pbar:
        # # The case when min/max are tensors: not supported until torch 1.9.1
        # q.data = torch.clamp(q.data, -qbound, qbound)
        q.data = torch.max(torch.min(q.data, qbound), -qbound)
        Iand_p = torch.matmul(reward2Iand, 0.5 * (rewards + q) + p)
        Ior_p = torch.matmul(reward2Ior, 0.5 * (rewards + q) - p)

        if and_10dim != None:
            and_10dim = torch.cat((and_10dim, Iand_p.clone()[q_indices[:10]].reshape(-1,1)), dim = 1)
        else:
            and_10dim = Iand_p.clone()[q_indices[:10]].reshape(-1,1)

        if or_10dim != None:
            or_10dim = torch.cat((or_10dim, Ior_p.clone()[q_indices[:10]].reshape(-1,1)), dim = 1)
        else:
            or_10dim = Ior_p.clone()[q_indices[:10]].reshape(-1,1)


        if loss_type == "l1":
            loss = torch.sum(torch.abs(Iand_p)) + torch.sum(torch.abs(Ior_p))  # 02-27:     .
            losses["loss"].append(loss.item())
        elif loss_type.startswith("l1_on"):
            loss = l1_on_given_dim(torch.cat([Iand_p, Ior_p]), indices=noisy_pattern_indices)
            losses["loss"].append(loss.item())
            losses["noise_ratio"].append(loss.item() / torch.sum(torch.abs(torch.cat([Iand_p, Ior_p]))).item())
        else:
            raise NotImplementedError(f"Loss type {loss_type} unrecognized.")

        if it + 1 < niter:
            optimizer.zero_grad()
            # change 使用固定的学习率
            # optimizer.param_groups[0]["lr"] = eta_list[it]
            loss.backward()
            optimizer.step()

        if it in ckpt_id_list:
            progresses["I_and"].append(Iand_p.detach().cpu().numpy())
            progresses["I_or"].append(Ior_p.detach().cpu().numpy())
            pbar.set_postfix_str(f"loss={loss.item():.4f}")

        losses["q_norm_L1"].append(torch.norm(q.data,p=1).item())
        losses["q_mean"].append(torch.mean(q.data).item())

        if q_10dim != None:
            q_10dim = torch.cat((q_10dim, q.clone()[q_indices[:10]].reshape(-1,1)), dim = 1)
        else:
            q_10dim = q.clone()[q_indices[:10]].reshape(-1,1)

        if p_10dim != None:
            p_10dim = torch.cat((p_10dim, p.clone()[q_indices[:10]].reshape(-1,1)), dim = 1)
        else:
            p_10dim = p.clone()[q_indices[:10]].reshape(-1,1)

        if q_tricks: # todo: 这个q_tricks真的有必要加吗？
            q.data = q.data - torch.mean(q.data)

    return p.detach(), q.detach(), losses, progresses, q_10dim


def _train_q(
        rewards: torch.Tensor,
        loss_type: str,
        lr: float,
        niter: int,
        qbound: Union[float, torch.Tensor],
        q_tricks: bool,
        reward2Iand: torch.Tensor = None,
        # reward2Ior: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
    device = rewards.device
    n_dim = int(np.log2(rewards.numel()))
    if reward2Iand is None:
        reward2Iand = get_reward2Iand_mat(n_dim).to(device)
    # if reward2Ior is None:
    #     reward2Ior = get_reward2Ior_mat(n_dim).to(device)

    log_lr = np.log10(lr)
    eta_list = np.logspace(log_lr, log_lr - 1, niter)
    q_10dim = None

    # p = torch.zeros_like(rewards).requires_grad_(True)
    q = torch.zeros_like(rewards).requires_grad_(True)
    optimizer = optim.SGD([q], lr=0.0, momentum=0.9)

    if loss_type == "l1":
        losses = {"loss": [], "q_norm_L1":[], "q_mean":[]}
    else:
        raise NotImplementedError(f"Loss type {loss_type} unrecognized.")

    progresses = {"I_and": []}
    ckpt_id_list = generate_ckpt_id_list(niter=niter, nckpt=20)

    pbar = tqdm(range(niter), desc="Optimizing pq", ncols=100)
    for it in pbar:
        # # The case when min/max are tensors: not supported until torch 1.9.1
        # q.data = torch.clamp(q.data, -qbound, qbound)
        q.data = torch.max(torch.min(q.data, qbound), -qbound)
        Iand_p = torch.matmul(reward2Iand, rewards + q)
        # Ior_p = torch.matmul(reward2Ior, 0.5 * (rewards + q) - p)

        if loss_type == "l1":
            loss = torch.sum(torch.abs(Iand_p)) # + torch.sum(torch.abs(Ior_p))  # 02-27: L1 penalty.
            losses["loss"].append(loss.item())
        else:
            raise NotImplementedError(f"Loss type {loss_type} unrecognized.")

        if it + 1 < niter:
            optimizer.zero_grad()
            optimizer.param_groups[0]["lr"] = eta_list[it]
            loss.backward()
            optimizer.step()

        if it in ckpt_id_list:
            progresses["I_and"].append(Iand_p.detach().cpu().numpy())
            # progresses["I_or"].append(Ior_p.detach().cpu().numpy())
            pbar.set_postfix_str(f"loss={loss.item():.4f}")

        losses["q_norm_L1"].append(torch.norm(q.data,p=1).item())
        losses["q_mean"].append(torch.mean(q.data).item())

        if q_10dim != None:
            q_10dim = torch.cat((q_10dim, q.clone()[:10].reshape(-1,1)), dim = 1)
        else:
            q_10dim = q.clone()[:10].reshape(-1,1)

        if q_tricks:
            q.data = q.data - torch.mean(q.data)

    return q.detach(), losses, progresses, q_10dim


def _train_p_q_piece_wise(
        rewards: torch.Tensor,
        loss_type: str,
        lr: float,
        niter: int,
        qbound: Union[float, torch.Tensor],
        q_tricks: bool,
        reward2Iand: torch.Tensor = None,
        reward2Ior: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
    device = rewards.device
    n_dim = int(np.log2(rewards.numel()))
    if reward2Iand is None:
        reward2Iand = get_reward2Iand_mat(n_dim).to(device)
    if reward2Ior is None:
        reward2Ior = get_reward2Ior_mat(n_dim).to(device)

    niter *= 2

    log_lr = np.log10(lr)
    eta_list = np.logspace(log_lr, log_lr - 1, niter)
    q_10dim = None

    p = torch.zeros_like(rewards).requires_grad_(True)
    q = torch.zeros_like(rewards).requires_grad_(True)
    optimizer = optim.SGD([p, q], lr=0.0, momentum=0.9)

    if loss_type == "l1":
        ratio = 1
        Iand_p = torch.matmul(reward2Iand, 0.5 * rewards + p)
        Ior_p = torch.matmul(reward2Ior, 0.5 * rewards - p)
        num_noisy_pattern = int(ratio * (Iand_p.shape[0] + Ior_p.shape[0]))
        print("# noisy patterns", num_noisy_pattern)
        noisy_pattern_indices = torch.argsort(torch.abs(torch.cat([Iand_p, Ior_p]))).tolist()[:num_noisy_pattern]
        losses = {"loss": [], "noise_ratio": [], "q_norm_L1":[], "q_mean":[]}
    elif loss_type.startswith("l1_on"):
        ratio = float(loss_type.split("_")[-1])
        Iand_p = torch.matmul(reward2Iand, 0.5 * rewards + p)
        Ior_p = torch.matmul(reward2Ior, 0.5 * rewards - p)
        num_noisy_pattern = int(ratio * (Iand_p.shape[0] + Ior_p.shape[0]))
        print("# noisy patterns", num_noisy_pattern)
        noisy_pattern_indices = torch.argsort(torch.abs(torch.cat([Iand_p, Ior_p]))).tolist()[:num_noisy_pattern]
        losses = {"loss": [], "noise_ratio": [], "q_norm_L1":[], "q_mean":[]}
    else:
        raise NotImplementedError(f"Loss type {loss_type} unrecognized.")

    progresses = {"I_and": [], "I_or": []}
    ckpt_id_list = generate_ckpt_id_list(niter=niter, nckpt=20)

    pbar = tqdm(range(niter), desc="Optimizing pq", ncols=100)
    for it in pbar:
        # # The case when min/max are tensors: not supported until torch 1.9.1
        # q.data = torch.clamp(q.data, -qbound, qbound)
        q.data = torch.max(torch.min(q.data, qbound), -qbound)
        Iand_p = torch.matmul(reward2Iand, 0.5 * (rewards + q) + p)
        Ior_p = torch.matmul(reward2Ior, 0.5 * (rewards + q) - p)


        if it < niter / 2:
            num_noisy_pattern = int(1 * (Iand_p.shape[0] + Ior_p.shape[0]))
            noisy_pattern_indices = torch.argsort(torch.abs(torch.cat([Iand_p, Ior_p]))).tolist()[:num_noisy_pattern]
        else:
            num_noisy_pattern = int(0.95 * (Iand_p.shape[0] + Ior_p.shape[0]))
            noisy_pattern_indices = torch.argsort(torch.abs(torch.cat([Iand_p, Ior_p]))).tolist()[:num_noisy_pattern]

        loss = l1_on_given_dim(torch.cat([Iand_p, Ior_p]), indices=noisy_pattern_indices)
        losses["loss"].append(loss.item())
        losses["noise_ratio"].append(loss.item() / torch.sum(torch.abs(torch.cat([Iand_p, Ior_p]))).item())


        if it + 1 < niter:
            optimizer.zero_grad()
            optimizer.param_groups[0]["lr"] = eta_list[it]
            loss.backward()
            optimizer.step()

        if it in ckpt_id_list:
            progresses["I_and"].append(Iand_p.detach().cpu().numpy())
            progresses["I_or"].append(Ior_p.detach().cpu().numpy())
            pbar.set_postfix_str(f"loss={loss.item():.4f}")

        losses["q_norm_L1"].append(torch.norm(q.data,p=1).item())
        losses["q_mean"].append(torch.mean(q.data).item())

        if q_10dim != None:
            q_10dim = torch.cat((q_10dim, q.clone()[:10].reshape(-1,1)), dim = 1)
        else:
            q_10dim = q.clone()[:10].reshape(-1,1)

        if q_tricks:
            q.data = q.data - torch.mean(q.data)

    return p.detach(), q.detach(), losses, progresses, q_10dim


def l1_on_given_dim(vector: torch.Tensor, indices: List) -> torch.Tensor:
    assert len(vector.shape) == 1
    strength = torch.abs(vector)
    return torch.sum(strength[indices])


def generate_ckpt_id_list(niter: int, nckpt: int) -> List:
    ckpt_id_list = list(range(niter))[::max(1, niter // nckpt)]
    # force the last iteration to be a checkpoint
    if niter - 1 not in ckpt_id_list:
        ckpt_id_list.append(niter - 1)
    return ckpt_id_list
