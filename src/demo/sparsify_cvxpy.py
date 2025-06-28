import os
import sys
sys.path.append(os.getcwd())
from utils.global_const import *
# Set the cache directory for huggingface hub. [Important!] It should be before the import of transformers
os.environ["HF_HOME"] = HF_HOME

import numpy as np
import cvxpy as cp
import torch
from interaction.harsanyi.and_or_harsanyi_utils import get_reward2Iand_mat, get_reward2Ior_mat
from interaction.harsanyi.set_utils import generate_all_masks
from scipy.special import comb
from utils import get_dataset_nlp, load_json
from transformers import AutoTokenizer
from interaction.player import get_player_words_from_ids
from interaction.harsanyi.plot import plot_interaction_progress, plot_multi_line_chart
import pandas as pd
from interaction.harsanyi.calculate import log_interaction
from models.nlp import Qwen2TokenizerModified

FONT = 20


def get_penalty_term(q: cp.Variable, v_size: int, qscale: str):
    n_dim_ = int(np.log2(v_size))
    player_masks_ = np.array(generate_all_masks(length=n_dim_, sort_type="order"), dtype=bool)

    if qscale == "uniform":
        weighting_matrix = np.eye(v_size)
        constraint_term = q.T @ weighting_matrix @ q
    elif qscale == "shapley":
        weighting_matrix = np.zeros((v_size, v_size))
        for i in range(v_size):
            order = np.sum(player_masks_[i])
            if order != 0 and order != n_dim:
                weighting_matrix[i, i] = (n_dim - 1) / (order * (n_dim - order) * comb(N=n_dim, k=order, exact=True))
        constraint_term = q.T @ weighting_matrix @ q
    else:
        raise NotImplementedError(f"qscale [{qscale}] is not implemented.")
    return constraint_term


def solve_pq_relax(reward2Iand, reward2Ior, rewards, qscale, lambda_reg, loss="l1", delta=1.0, solver="MOSEK"):
    assert qscale in ["uniform", "shapley"]
    assert loss in ["l1", "huber"]

    v_size_ = rewards.shape[0]

    # Define the variables
    print("Define the variables...")
    q = cp.Variable(v_size_)
    p = cp.Variable(v_size_)
    print("done.")

    if loss == "l1":
        I_and_norm = cp.norm(reward2Iand @ (0.5 * (rewards + q) + p), 1)
        I_or_norm = cp.norm(reward2Ior @ (0.5 * (rewards + q) - p), 1)
    elif loss == "huber":
        I_and_norm = cp.sum(cp.huber(reward2Iand @ (0.5 * (rewards + q) + p), M=delta)) / (2.0 * delta)
        I_or_norm = cp.sum(cp.huber(reward2Ior @ (0.5 * (rewards + q) - p), M=delta)) / (2.0 * delta)
    else:
        raise NotImplementedError(f"loss [{loss}] is not implemented.")
    print("done.")

    print("getting constraint term...")
    penalty_term = get_penalty_term(q, v_size_, qscale)
    print("done.")
    objective = cp.Minimize(I_and_norm + I_or_norm + lambda_reg * penalty_term)

    if qscale == "uniform":
        constraints = []
    elif qscale == "shapley":
        # contrain the first and last element of q to be 0
        constraints = [q[0] == 0, q[-1] == 0]
    else:
        raise NotImplementedError(f"qscale [{qscale}] is not implemented.")

    # Solve the problem
    print("Solving the problem...")
    problem = cp.Problem(objective, constraints)
    if solver == "MOSEK":
        kwargs = {"accept_unknown": True}
    else:
        kwargs = {}
    problem.solve(verbose=True, solver=solver, **kwargs)
    print("done.")

    # Get the solution
    if problem.status == cp.OPTIMAL:
        print("Optimal p:", p.value)
        print("Optimal q:", q.value)
        print("Optimal objective value:", problem.value)
    else:
        print("Optimization failed:", problem.status)

    return p.value, q.value, problem.value, problem.status



def solve_pq_constraint(reward2Iand, reward2Ior, rewards, q_bound, loss="l1", delta=1.0, solver="MOSEK"):
    assert loss in ["l1", "huber"]

    v_size_ = rewards.shape[0]

    # Define the variables
    print("Define the variables...")
    q = cp.Variable(v_size_)
    p = cp.Variable(v_size_)
    print("done.")

    if loss == "l1":
        I_and_norm = cp.norm(reward2Iand @ (0.5 * (rewards + q) + p), 1)
        I_or_norm = cp.norm(reward2Ior @ (0.5 * (rewards + q) - p), 1)
    elif loss == "huber":
        I_and_norm = cp.sum(cp.huber(reward2Iand @ (0.5 * (rewards + q) + p), M=delta)) / (2.0 * delta)
        I_or_norm = cp.sum(cp.huber(reward2Ior @ (0.5 * (rewards + q) - p), M=delta)) / (2.0 * delta)
    else:
        raise NotImplementedError(f"loss [{loss}] is not implemented.")
    print("done.")

    objective = cp.Minimize(I_and_norm + I_or_norm)
    constraints = [cp.norm_inf(q) <= q_bound]

    # Solve the problem
    print("Solving the problem...")
    problem = cp.Problem(objective, constraints)
    if solver == "MOSEK":
        kwargs = {"accept_unknown": True}
    else:
        kwargs = {}
    problem.solve(verbose=True, solver=solver, **kwargs)
    print("done.")

    # Get the solution
    if problem.status == cp.OPTIMAL:
        print("Optimal p:", p.value)
        print("Optimal q:", q.value)
        print("Optimal objective value:", problem.value)
    else:
        print("Optimization failed:", problem.status)

    return p.value, q.value, problem.value, problem.status


def plot_interactions_compare(save_folder, I_and, I_or, I_and_sparsify, I_or_sparsify):
    plot_interaction_progress(
        interaction=[I_and, I_and_sparsify],
        save_path=os.path.join(save_folder, f"I_and_compare.png"),
        title="I_and",
        cmap_name="bwr",
        hline_color="black",
        font=FONT
    )

    plot_interaction_progress(
        interaction=[I_or, I_or_sparsify],
        save_path=os.path.join(save_folder, f"I_or_compare.png"),
        title="I_or",
        cmap_name="bwr",
        hline_color="black",
        font = FONT
    )


def plot_v_compare(save_folder, v_, p_, q_, sort_idx):
    index = np.arange(v_.shape[0])

    # plot universal matching (sort by order)
    plot_multi_line_chart(
        data=pd.DataFrame({"v": v_,
                           "v+q": v_ + q_,
                           "index": index}),
        xlabel="index",
        ylabel=["v", "v+q"],
        title="universal matching (sort by order)",
        save_folder=save_folder,
        save_name=f"universal_matching_sort_by_order",
        font=FONT
    )

    # plot universal matching (sort by v's value)
    plot_multi_line_chart(
        data=pd.DataFrame({"v": v_[sort_idx],
                           "v+q": (v_ + q_)[sort_idx],
                           "index": index}),
        xlabel="index",
        ylabel=["v", "v+q"],
        title="universal matching (sort by v's value)",
        save_folder=save_folder,
        save_name=f"universal_matching_sort_by_v_value",
        font=FONT
    )

    # plot p (sort by order)
    plot_multi_line_chart(
        data=pd.DataFrame({"0.5v": 0.5 * v_,
                           "0.5v+p": 0.5 * v_ + p_,
                           "0.5v-p": 0.5 * v_ - p_,
                           "index": index}),
        xlabel="index",
        ylabel=["0.5v", "0.5v+p", "0.5v-p"],
        title="visualize p (sort by order)",
        save_folder=save_folder,
        save_name=f"visualize_p_sort_by_order",
        font=FONT
    )

    # plot p (sort by v's value)
    sort_idx = np.argsort(v)
    plot_multi_line_chart(
        data=pd.DataFrame({"0.5v": 0.5 * v_[sort_idx],
                           "0.5v+p": (0.5 * v_ + p_)[sort_idx],
                           "0.5v-p": (0.5 * v_ - p_)[sort_idx],
                           "index": index}),
        xlabel="index",
        ylabel=["0.5v", "0.5v+p", "0.5v-p"],
        title="visualize p (sort by v's value)",
        save_folder=save_folder,
        save_name=f"visualize_p_sort_by_v_value",
        font=FONT
    )


def reorganize_and_or_interactions(I_and: np.ndarray, I_or: np.ndarray, player_masks: np.ndarray):
    """
    To combine the single-order, zero-order components in and-or interactions together
    """
    with torch.no_grad():
        I_and_ = I_and.copy()
        I_or_ = I_or.copy()

        # todo: 20250131 这个reorganize可能是有问题的，因为1阶与交互和1阶或交互表示的含义不一样。v(empty)可以合并，但1阶的不应该合并
        # comb_indices = np.sum(player_masks, axis=1) <= 1
        comb_indices = np.sum(player_masks, axis=1) == 0

        I_and_[comb_indices] = I_and_[comb_indices] + I_or_[comb_indices]
        I_or_[comb_indices] = 0

    return I_and_, I_or_


if __name__ == '__main__':

    ### model: bertweet
    ### dataset: custom-imdb-for-bertweet-nips2024-ucb
    # tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
    # dataset_name = "custom-imdb-for-bertweet-nips2024-ucb"
    # root_path_list = [
    #     "../results/20250116_examine_pq_optimization/result/dataset=custom-imdb-for-bertweet-nips2024-ucb-test_model=BERTweet#pretrain_seed=0/players=%s_mode=pq_lbl=correct_baseline=unk_bg=ori#_loss=%s_optim=sgd_autolr=v1_mom=0.999_niters=50000_qcoef=0.05"
    # ]
    # # player_folder_list = ["players-manual", "players-at-most-10"]
    # player_folder_list = ["players-manual"]
    # sample_idx_list = [0, 1, 2, 3,4,5,6, 7, 8, 9, 10]


    ### model: llama-7b
    ### dataset: custom-generation-test
    # tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    # dataset_name = "custom-generation-test"
    # root_path_list = [
    #     "../results/20250118_try_generation_test/result/dataset=custom-generation-test-test_model=llama-7b#pretrain_seed=0/players=%s_mode=pq_lbl=predict_baseline=unk_bg=ori#_loss=%s_optim=sgd_autolr=v1_mom=0.999_niters=50000_qcoef=0.05",
    #     "../results/20250118_try_generation_test/result/dataset=custom-generation-test-test_model=llama-7b#pretrain_seed=0/players=%s_mode=pq_v=gt_lbl=predict_baseline=unk_bg=ori#_loss=%s_optim=sgd_autolr=v1_mom=0.999_niters=50000_qcoef=0.05",
    #     "../results/20250118_try_generation_test/result/dataset=custom-generation-test-test_model=llama-7b#pretrain_seed=0/players=%s_mode=pq_v=gt-log-odds-sample=1000_lbl=predict_baseline=unk_bg=ori#_loss=%s_optim=sgd_autolr=v1_mom=0.999_niters=50000_qcoef=0.05"
    # ]
    # player_folder_list = ["players-llama-manual"]
    # sample_idx_list = [0,2,3]


    ### model: opt-1.3b
    ### dataset: custom-generation-test
    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", add_prefix_space=True)
    # dataset_name = "custom-generation-test"
    # root_path_list = [
    #     "../results/20250118_try_generation_test/result/dataset=custom-generation-test-test_model=OPT-1.3b#pretrain_seed=0/players=%s_mode=pq_lbl=predict_baseline=pad_bg=ori#_loss=%s_optim=sgd_autolr=v1_mom=0.999_niters=50000_qcoef=0.05",
    #     "../results/20250118_try_generation_test/result/dataset=custom-generation-test-test_model=OPT-1.3b#pretrain_seed=0/players=%s_mode=pq_v=gt_lbl=predict_baseline=pad_bg=ori#_loss=%s_optim=sgd_autolr=v1_mom=0.999_niters=50000_qcoef=0.05",
    #     "../results/20250118_try_generation_test/result/dataset=custom-generation-test-test_model=OPT-1.3b#pretrain_seed=0/players=%s_mode=pq_v=gt-log-odds-sample=1000_lbl=predict_baseline=pad_bg=ori#_loss=%s_optim=sgd_autolr=v1_mom=0.999_niters=50000_qcoef=0.05"
    # ]
    # player_folder_list = ["players-opt-manual"]
    # sample_idx_list = [0, 2, 3]


    ### model: pythia-suite
    ### dataset: custom-generation-test
    # # model_name = "410m#pretrain"
    # # model_name = "1.4b#pretrain"
    # model_name = "6.9b#pretrain"
    # # model_name = "12b#pretrain"
    # tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-{model_name.split('#')[0]}", add_prefix_space=True)
    # dataset_name = "custom-generation-test"
    # root_path_list = [
    #     f"../results/20250118_try_generation_test/result/dataset=custom-generation-test-test_model=pythia-{model_name}_seed=0/players=%s_mode=pq_lbl=predict_baseline=unk_bg=ori#_loss=%s_optim=sgd_autolr=v1_mom=0.999_niters=50000_qcoef=0.05",
    # ]
    # player_folder_list = ["players-pythia-manual"]
    # sample_idx_list = [0,2,3]


    ### model: pythia-suite
    ### dataset: custom-squad-from-sw
    # # model_name = "410m#pretrain"
    # # model_name = "1.4b#pretrain"
    # # model_name = "6.9b#pretrain"
    # model_name = "12b#pretrain"
    # tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-{model_name.split('#')[0]}", add_prefix_space=True)
    # dataset_name = "custom-squad-from-sw"
    # root_path_list = [
    #     f"../results/20250118_try_squad_from_sw/result/dataset=custom-squad-from-sw-test_model=pythia-{model_name}_seed=0/players=%s_mode=pq_lbl=predict_baseline=unk_bg=ori#_loss=%s_optim=sgd_autolr=v1_mom=0.999_niters=50000_qcoef=0.05",
    # ]
    # player_folder_list = ["players-pythia-from-sw"]
    # sample_idx_list = np.arange(0, 100)


    ### model: pythia-suite
    ### dataset: custom-squad-v1-20250126-val
    # model_name = "14m#pretrain"
    # model_name = "70m#pretrain"
    # model_name = "70m-deduped#pretrain"
    # model_name = "160m#pretrain"
    # model_name = "160m-deduped#pretrain"
    # model_name = "410m#pretrain"
    # model_name = "410m-deduped#pretrain"
    # model_name = "1b#pretrain"
    # model_name = "1b-deduped#pretrain"
    # model_name = "1.4b#pretrain"
    # model_name = "1.4b#pretrain_revision=step1000"
    # model_name = "1.4b#pretrain_revision=step1"
    # model_name = "1.4b#pretrain_revision=step0"
    # model_name = "1.4b-deduped#pretrain"
    # model_name = "1.4b-deduped#pretrain_revision=step1000"
    # model_name = "1.4b-deduped#pretrain_revision=step1"
    # model_name = "2.8b#pretrain"
    # model_name = "6.9b#pretrain"
    # model_name = "6.9b-deduped#pretrain"
    # model_name = "12b#pretrain"
    # model_name = "12b-deduped#pretrain"
    # tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-{model_name.split('#')[0]}", add_prefix_space=True)
    # dataset_name = "custom-squad-v1-20250126-val"
    # root_path_list = [
    #     # f"../results/20250126_try_squad_v1/result/dataset=custom-squad-v1-20250126-val-test_model=pythia-{model_name}_seed=0/players=%s_mode=pq_lbl=predict_baseline=unk_bg=ori#_loss=%s_optim=sgd_autolr=v1_mom=0.999_niters=50000_qcoef=0.05",
    #     # f"../results/20250126_try_squad_v1_wo_reorg/result/dataset=custom-squad-v1-20250126-val-test_model=pythia-{model_name}_seed=0/players=%s_mode=pq_lbl=predict_baseline=unk_bg=ori#_loss=%s_optim=sgd_autolr=v1_mom=0.999_niters=50000_qcoef=0.05",
    #     f"../results/20250126_try_squad_v1_use_mean_q/result/dataset=custom-squad-v1-20250126-val-test_model=pythia-{model_name}_seed=0/players=%s_mode=pq_lbl=predict_baseline=unk_bg=ori#_loss=%s_optim=sgd_autolr=v1_mom=0.999_niters=50000_qcoef=0.05_qstd=mean-vN-v0",
    # ]
    # player_folder_list = ["players-pythia"]
    # sample_idx_list = np.arange(0, 100)


    ### model: qwen2.5
    ### dataset: custom-squad-v2-20250202-val
    # model_name = "0.5B#pretrain"
    # model_name = "1.5B#pretrain"
    # model_name = "3B#pretrain"
    # model_name = "7B#pretrain"
    # model_name = "14B#pretrain"
    # tokenizer = Qwen2TokenizerModified.from_pretrained(f"Qwen/Qwen2.5-{model_name.split('#')[0]}", add_prefix_space=True)
    # dataset_name = "custom-squad-v2-20250202-val"
    # root_path_list = [
    #     f"../results/20250203_try_squad_v2_use_mean_q/result/dataset=custom-squad-v2-20250202-val-test_model=qwen2.5-{model_name.lower()}_seed=0/players=%s_mode=pq_lbl=predict_baseline=pad_bg=ori#_loss=%s_optim=sgd_autolr=v1_mom=0.999_niters=50000_qcoef=0.05_qstd=mean-vN-v0",
    # ]
    # player_folder_list = ["players-qwen"]
    # sample_idx_list = np.arange(0, 100)


    ### model: qwen2.5
    ### dataset: custom-cn-us-from-cqa
    # model_name = "0.5B#pretrain"
    # model_name = "1.5B#pretrain"
    # model_name = "3B#pretrain"
    model_name = "7B#pretrain"
    # model_name = "14B#pretrain"
    tokenizer = Qwen2TokenizerModified.from_pretrained(f"Qwen/Qwen2.5-{model_name.split('#')[0]}",
                                                       add_prefix_space=True)
    dataset_name = "custom-cn-us-from-cqa"
    root_path_list = [
        f"../results/20250207_try_cn_us_from_cqa/result/dataset=custom-cn-us-from-cqa-test_model=qwen2.5-{model_name.lower()}_seed=0/players=%s_mode=pq_lbl=predict_baseline=pad_bg=ori#_loss=%s_optim=sgd_autolr=v1_mom=0.999_niters=50000_qcoef=0.004_qstd=mean-vN-v0",
        f"../results/20250207_try_cn_us_from_cqa/result/dataset=custom-cn-us-from-cqa-test_model=qwen2.5-{model_name.lower()}_seed=0/players=%s_mode=pq_lbl=predict_baseline=pad_bg=ori#_loss=%s_optim=sgd_autolr=v1_mom=0.999_niters=50000_qcoef=0.05_qstd=mean-vN-v0",
    ]
    player_folder_list = ["players-qwen-from-cqa"]
    sample_idx_list = np.arange(0, 78)


    # =============================================================================

    dataset = get_dataset_nlp(dataset_name=dataset_name,
                              mode="eval",
                              tokenizer=tokenizer,
                              batch_size=1,
                              data_path=f"../datasets/{dataset_name}")
    dataset.load_data()

    # todo: 现在的q_bound是直接从文件中读取的，所以下面的q_coef和q_std都是无效的
    # q_coef = 0.05
    # # q_std = "vN-v0"
    # q_std = "mean-vN-v0"

    # todo
    solver = 'MOSEK'

    # todo:
    huber_delta = 1e-2

    for root_path in root_path_list:
        for player_folder in player_folder_list:
            for sample_idx in sample_idx_list:
                for loss in ["l1"]:
                    loss_note = ("l1" if loss == "l1" else f"huber-{huber_delta:.1e}")
                    sample_path = os.path.join(root_path % (player_folder, loss_note), f"data/sample{sample_idx}")

                    # sentence = np.array(dataset.dataset["text"])[sample_idx]
                    input_ids = dataset.dataset["input_ids"][sample_idx] # tensor
                    # attention_mask = dataset.dataset["attention_mask"][sample_idx] # tensor

                    player_ids_dict = load_json(f"../players/{dataset_name}/{player_folder}/player_ids_from_word.json")
                    player_ids = player_ids_dict[str(sample_idx)]
                    player_descriptions = get_player_words_from_ids(tokenizer, input_ids.squeeze(), player_ids)

                    v = np.load(os.path.join(sample_path, "rewards_minus_v0.npy"))
                    player_masks = np.load(os.path.join(sample_path, "player_masks.npy"))

                    v_size = v.shape[0]
                    n_dim = int(np.log2(v.shape[0]))
                    assert 2 ** n_dim == v_size

                    A = get_reward2Iand_mat(n_dim=n_dim, sort_type="order").cpu().numpy()
                    B = get_reward2Ior_mat(n_dim=n_dim, sort_type="order").cpu().numpy()

                    I_and = A @ (0.5 * v)
                    I_or = B @ (0.5 * v)

                    ### get q_bound
                    # if q_std == "vN-v0":
                    #     q_bound = q_coef * np.abs(v[-1] - v[0])
                    # elif q_std == "mean-vN-v0":
                    #     mean_of_vN_v0 =
                    #     q_bound = q_coef * mean_of_vN_v0
                    # else:
                    #     raise NotImplementedError(f"q_std [{q_std}] is not implemented.")
                    q_bound = np.load(os.path.join(sample_path, "after_sparsify/q_bound.npy"))
                    print("q_bound: ", q_bound)

                    # plot v (sort by order)
                    plot_multi_line_chart(
                        data=pd.DataFrame({"v": v, "index": np.arange(v_size)}),
                        xlabel="index",
                        ylabel=["v"],
                        title="v(S) (sort by order)",
                        save_folder=sample_path,
                        save_name=f"vS_sort_by_order",
                        font=FONT
                    )

                    # plot v (sort by value)
                    sort_idx = np.argsort(v)

                    plot_multi_line_chart(
                        data=pd.DataFrame({"v": v[sort_idx], "index": np.arange(v_size)}),
                        xlabel="index",
                        ylabel=["v"],
                        title="v(S) (sort by value)",
                        save_folder=sample_path,
                        save_name=f"vS_sort_by_value",
                        font=FONT
                    )


                    print("========= Solving p and q with constraint... ========")
                    save_root = os.path.join(sample_path, f"try_pq_constraint_cvxpy_loss={loss_note}_solver={solver}")
                    os.makedirs(save_root, exist_ok=True)
                    p, q, loss_value, problem_status = solve_pq_constraint(reward2Iand=A, reward2Ior=B, rewards=v, q_bound=q_bound,
                                                  loss=loss, delta=huber_delta, solver=solver)
                    I_and_sparsify = A @ (0.5 * (v + q) + p)
                    I_or_sparsify = B @ (0.5 * (v + q) - p)
                    with open(os.path.join(save_root, "problem_status.txt"), "w") as f:
                        f.write(f"problem_status: {problem_status}\nfinal_loss_value: {loss_value}")

                    # First reorganize interactions: combine the single-order, zero-order components in and-or interactions together
                    # todo: 20250131 这个reorganize可能是有问题的，因为1阶与交互和1阶或交互表示的含义不一样。v(empty)可以合并，但1阶的不应该合并
                    I_and_sparsify, I_or_sparsify = reorganize_and_or_interactions(I_and_sparsify,
                                                                                   I_or_sparsify,
                                                                                   player_masks)
                    np.save(os.path.join(save_root, "I_and.npy"), I_and_sparsify)
                    np.save(os.path.join(save_root, "I_or.npy"), I_or_sparsify)
                    np.save(os.path.join(save_root, "p.npy"), p)
                    np.save(os.path.join(save_root, "q.npy"), q)

                    plot_interactions_compare(save_root, I_and=I_and, I_or=I_or, I_and_sparsify=I_and_sparsify,
                                              I_or_sparsify=I_or_sparsify)
                    plot_v_compare(save_root, v, p, q, sort_idx)
                    log_interaction(save_path=save_root,
                                    player_ids=player_ids,
                                    player_masks=player_masks,
                                    I_and=I_and_sparsify,
                                    I_or=I_or_sparsify,
                                    player_descriptions=player_descriptions)



                    # for qscale, lambda_reg in [("uniform", 1.0), ("shapley", 10.0), ("shapley", 100.0)]:
                    #     print("========= Solving p and q with penalty term... ========")
                    #     print(f"qscale={qscale}, lambda_reg={lambda_reg}")
                    #
                    #     save_root = os.path.join(sample_path, f"try_pq_relaxation_cvxpy_loss={loss_note}_solver={solver}",
                    #                              f"qscale={qscale}_lam={lambda_reg}")
                    #     os.makedirs(save_root, exist_ok=True)
                    #
                    #     print("Solving p and q...")
                    #     p, q, loss_value, problem_status = solve_pq_relax(reward2Iand=A, reward2Ior=B, rewards=v, qscale=qscale,
                    #                              lambda_reg=lambda_reg, loss=loss, delta=huber_delta, solver=solver)
                    #     I_and_sparsify = A @ (0.5 * (v + q) + p)
                    #     I_or_sparsify = B @ (0.5 * (v + q) - p)
                    #     with open(os.path.join(save_root, "problem_status.txt"), "w") as f:
                    #         f.write(f"problem_status: {problem_status}\nfinal_loss_value: {loss_value}")
                    #
                    #     # First reorganize interactions: combine the single-order, zero-order components in and-or interactions together
                    #     I_and_sparsify, I_or_sparsify = reorganize_and_or_interactions(I_and_sparsify,
                    #                                                                    I_or_sparsify,
                    #                                                                    player_masks)
                    #     np.save(os.path.join(save_root, "I_and.npy"), I_and_sparsify)
                    #     np.save(os.path.join(save_root, "I_or.npy"), I_or_sparsify)
                    #     np.save(os.path.join(save_root, "p.npy"), p)
                    #     np.save(os.path.join(save_root, "q.npy"), q)
                    #
                    #     plot_interactions_compare(save_root, I_and=I_and, I_or=I_or, I_and_sparsify=I_and_sparsify,
                    #                               I_or_sparsify=I_or_sparsify)
                    #     plot_v_compare(save_root, v, p, q, sort_idx)
                    #     log_interaction(save_path=save_root,
                    #                     player_ids=player_ids,
                    #                     player_masks=player_masks,
                    #                     I_and=I_and_sparsify,
                    #                     I_or=I_or_sparsify,
                    #                     player_descriptions=player_descriptions)
                    #
                    #
