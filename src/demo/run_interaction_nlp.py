import torch

import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import os
import argparse

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interaction.player import *
from interaction.harsanyi import load_baseline_embeds, InteractionNLP
from utils import (set_seed, log_args, setup_path, setup_model_args_nlp, get_model_nlp, get_dataset_nlp,
                   load_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general settings
    parser.add_argument('--save_root', type=str, default="../results/20250101_nlp_interaction")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_type_name', type=str, default="auto",
                        choices=["auto", "fp16", "fp32", "fp64"])

    # model and dataset settings
    parser.add_argument("--model", default="Bert-base#pretrain", type=str,
                        help="the model to use")
    parser.add_argument("--dataset", default="SST-2", type=str,
                        help="set the dataset used")
    parser.add_argument('--data_path', default=None, type=str,
                        help="root folder for dataset / path to custom data file.")
    parser.add_argument('--batch_size', type=int, default=1) # when computing interactions, we usually load one sample at a time
    parser.add_argument("--data_split", default="test", type=str,
                        help="set the data split to run: train, test (val)")

    # settings for interaction calculation
    parser.add_argument("--interaction_type", default="harsanyi", type=str,
                        choices=["harsanyi", "shapley_taylor", "shapley_interaction_index", "re", "shapley"],
                        help="type of interaction to compute")
    parser.add_argument("--selected_dim", default="gt-log-odds", type=str,
                        choices=["gt", "gt-log-odds", "gt-log-odds-sample=1000"],
                        help="set the value function v() used in the interaction calculation.")
    parser.add_argument('--gt_type', type=str, default="predict",
                        choices=["predict", "correct"],  # todo: 如果要自己指定label，需要在custom dataset folder下面写一个labels.txt
                        help="whether to use the predicted label or the correct label for the value function v()")
    parser.add_argument("--player_path", default="../players/SST-2/players-20250104test", type=str,
                        help='the path of pre-sampled players and selected input sample indices')
    parser.add_argument('--baseline_type', type=str, default="pad",  # todo: 图像的baseline_type又不一样了，图像可以选择在中层遮挡？？要再加一个参数mask_layer？
                        choices=["learned", "pad", "unk", "bos", "eos", "mask"])
    parser.add_argument('--baseline_path', default='', type=str,
                        help="path for storing learned baseline values for masking"
                             "e.g., ../baseline_results/SST-2/baseline_embeds_list.npy")
    parser.add_argument('--background_type', type=str, default="mask",
                        choices=["mask", "ori"],
                        help="whether the background players are masked or original")
    parser.add_argument("--sort_type", default="order", type=str,
                        choices=["order", "binary"])
    parser.add_argument("--cal_batch_size", default=None, type=int,
                        help="batch size for computing forward passes on masked input samples")
    parser.add_argument("--verbose", default=1, type=int)

    # settings for sparsification
    # parser.add_argument('--sparse_mode', type=str, default="pq",
    #                     choices=["pq", "p", "q", "None"])
    # parser.add_argument("--loss", default="l1", type=str,
    #                     choices=["l1", "huber"])
    # parser.add_argument("--delta", default=1.0, type=float,
    #                     help="delta for huber loss")
    # parser.add_argument("--optimizer", default="sgd", type=str,
    #                     choices=["sgd", "adam"])
    # parser.add_argument('--lr', type=float, default=1e-5)
    # parser.add_argument('--auto_lr', type=str, default=None, choices=["v1"],
    #                     help="strategy to automatically set lr, will override the lr args")
    # parser.add_argument("--momentum", default=0.9, type=float)
    # parser.add_argument('--niters', type=int, default=20000)
    # parser.add_argument('--qcoef', type=float, default=0.02)
    # parser.add_argument("--qstd", default="vN-v0", type=str)
    # parser.add_argument("--qscale", default="uniform", type=str,
    #                     choices=["uniform", "shapley"])
    # parser.add_argument("--qtricks", default=0, type=int, choices=[0,1])
    # parser.add_argument("--piecewise", default=0, type=int, choices=[0,1])
    # parser.add_argument("--init_pq_path", default="", type=str,
    #                     help="path to the initial p and q values for sparsification")
    args = parser.parse_args()

    args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    assert args.batch_size == 1

    save_note = f"players={os.path.basename(args.player_path)}" \
                + (f"_v={args.selected_dim}" if args.selected_dim != "gt-log-odds" else "") \
                + f"_lbl={args.gt_type}" \
                + f"_baseline={args.baseline_type}" \
                + f"_bg={args.background_type}" \
                + (f"_sort={args.sort_type}" if args.sort_type != "order" else "") \
                + "#"  # a separator

    # if args.sparse_mode != "none":
    #     save_note += f"_loss={args.loss}" \
    #                  + (f"-{args.delta:.1e}" if args.loss == "huber" else "") \
    #                  + f"_optim={args.optimizer}" \
    #                  + (f"_autolr={args.auto_lr}" if args.auto_lr is not None else f"_lr={args.lr}") \
    #                  + (f"_mom={args.momentum}" if args.optimizer == "sgd" else "") \
    #                  + f"_niters={args.niters}" \
    #                  + f"_qcoef={args.qcoef}" \
    #                  + (f"_qstd={args.qstd}" if args.qstd != "vN-v0" else "") \
    #                  + (f"_qscale={args.qscale}" if args.qscale != "uniform" else "") \
    #                  + (f"_qtricks" if args.qtricks else "") \
    #                  + (f"_pw" if args.piecewise else "")

    # Set up the path for saving the results and logs
    # args.save_path_result, args.save_path_log
    setup_path(args, save_note)

    # Set up the model arguments
    setup_model_args_nlp(args)

    # Set up the random seed
    set_seed(args.seed)

    # get calculator (the model wrapper) and load parameters if needed
    calculator = get_model_nlp(args, mode="eval") # already to(device) and set model.eval()
    if args.data_type_name == "fp16":
        calculator = calculator.to(torch.float16)
    elif args.data_type_name == "fp32":
        calculator = calculator.to(torch.float32)
    elif args.data_type_name == "fp64":
        calculator = calculator.to(torch.float64)
    elif args.data_type_name == "auto":
        args.data_type_actual = calculator.model.dtype  # get the actual data type of the model, for logging purposes
        print(f"Auto data type is: {args.data_type_actual}")
    else:
        raise ValueError(f"Unknown data_type_name: {args.data_type_name}")

    # get the dataset
    dataset = get_dataset_nlp(dataset_name=args.dataset,
                              mode="eval",
                              tokenizer=calculator.tokenizer,
                              batch_size=args.batch_size, # restricted to 1
                              data_path=args.data_path) # can be None
    train_loader, test_loader = dataset.get_dataloader(shuffle_train=False)
    data_loader = train_loader if args.data_split == "train" else test_loader


    # todo: players文件夹的名字要不要显示在result文件夹里？
    player_ids_dict = load_json(os.path.join(args.player_path, "player_ids_from_word.json"))
    selected_sample_indices = player_ids_dict.keys() # todo: check 是list? 还是dict? 还是其他？
    selected_sample_indices = map(int, selected_sample_indices)
    print("selected_sample_indices", selected_sample_indices)

    if args.baseline_type == "learned":
        assert os.path.exists(args.baseline_path), f"Baseline path {args.baseline_path} does not exist."
        baseline_value = load_baseline_embeds(args.baseline_path)
        baseline_value = baseline_value.double() if args.data_type == "double" else baseline_value
    else:
        baseline_value = None

    # if args.qstd == "mean-vN-v0": # actually we use the mean of |vN-v0|, not the mean of vN-v0
    #     statistics_save_path = os.path.join(f"../saved_statistics/{args.dataset}-{args.data_split}/{args.model}",
    #                              f"player={os.path.basename(args.player_path)}_v={args.selected_dim}_lbl={args.gt_type}_baseline={args.baseline_type}_bg={args.background_type}" + (
    #                                  f"_sort={args.sort_type}" if args.sort_type != "order" else ""))
    #     mean_of_vN_v0 = np.load(os.path.join(statistics_save_path, "statistics/vN_v0_abs_mean.npy"))
    #     mean_of_vN_v0 = torch.from_numpy(mean_of_vN_v0).to(args.device)
    # else:
    #     mean_of_vN_v0 = None

    # prepare the config
    config = {
        "task": args.task,
        "data_type_name": args.data_type_name,
        "interaction_type": args.interaction_type,
        "selected_dim": args.selected_dim,
        "baseline_type": args.baseline_type,
        "gt_type": args.gt_type,
        "background_type": args.background_type,
        "sort_type": args.sort_type,
        "cal_batch_size": args.cal_batch_size,
        "verbose": args.verbose,
        "interaction_type": args.interaction_type,
        # Following are sparse configs
        # "sparse_mode": args.sparse_mode,
        # "loss": args.loss,
        # "delta": args.delta,
        # "optimizer": args.optimizer,
        # "lr": args.lr,
        # "auto_lr": args.auto_lr,
        # "momentum": args.momentum,
        # "niters": args.niters,
        # "qcoef": args.qcoef,
        # "qstd": args.qstd,
        # "qscale": args.qscale,
        # "qtricks": args.qtricks,
        # "piecewise": args.piecewise,
        # "init_pq_path": args.init_pq_path,
        # "mean_of_vN_v0": mean_of_vN_v0
    }

    # prepare the interaction runner (a wrapper class)
    interaction_nlp = InteractionNLP(calculator=calculator, config=config)

    # log everything after loading the model and data
    log_args(args.save_path_log, args)

    for i, data_tuple in enumerate(data_loader):
        # todo: 以及如果是用correct label，要不要剔除掉分类错误的样本 —— 这个要不先在结果里用log记录一下，之后再筛选

        # Todo: note that the following 'continue' statement is only valid when batch_size=1
        if i not in selected_sample_indices:
            continue

        if i >= 100: # todo: for debugging, remove this line
            continue

        set_seed(args.seed * 1000 + i) # set seed for each sample to ensure reproducibility

        save_path = os.path.join(args.save_path_result, f"sample{i}")
        sample_id = i
        player_ids = player_ids_dict[str(i)]
        print(f"sample_id: {sample_id}, player_ids: {player_ids}")
        assert isinstance(player_ids, List), f"player_ids should be a list, but got {type(player_ids)}"

        interaction_nlp(data_tuple, player_ids, save_path, sample_id, baseline_value)
