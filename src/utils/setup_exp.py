import re
from .tools import mkdir
import os
from .global_const import *

# Set the cache directory for huggingface hub. [Important!] It should be before the import of transformers
os.environ["HF_HOME"] = HF_HOME

from transformers import (BertForSequenceClassification, OPTForCausalLM, LlamaForCausalLM,
                          AutoModelForSequenceClassification, GPTNeoXForCausalLM, AutoModelForCausalLM,
                          Qwen2Tokenizer, AutoTokenizer)
from models.nlp import Qwen2TokenizerModified


def setup_path(args, save_note):
    """
    Setup the path for saving the results and logs
    """
    args.dataset_model = f"dataset={args.dataset}" \
                         + (f"-{args.data_split}" if args.data_split != "None" else "") \
                         + f"_model={args.model}" \
                         + f"_seed={args.seed}"
    args.save_path = os.path.join(args.save_root, "result", args.dataset_model, save_note) # result用来放各种数据，另外再开个analysis文件夹放分析结果
    args.save_path_result = os.path.join(args.save_path, "data")
    args.save_path_log = os.path.join(args.save_path, "log")
    mkdir(args.save_path_result)
    mkdir(args.save_path_log)


def parse_model_name(model_name: str):
    # use # to separate arch and extra config
    # extra config args are separated by "_"
    # E.g., "resnet20#lr=0.1_seed=1", "mlp#nlayer=5_width=1024_act=relu", "Bert-tiny#pretrain", "OPT-1.3b#pretrain",
    #   "llama-7b#pretrain", "pythia-70m#pretrain_revision=step1000" ...
    arch, extra_config_str = model_name.split("#")

    # parse extra config according to arch
    extra_config = {}

    # if arch == "mlp":
    #     nlayer, width, activation = extra_config_str.split("-")[:3]
    #     extra_config["nlayer"] = int(nlayer)
    #     extra_config["width"] = int(width)
    #     extra_config["act"] = activation

    extra_config_str_split = extra_config_str.split("_")
    # if any of the substrings contains "=", then it is a key-value pair, e.g., "revision=step1000"
    for s in extra_config_str_split:
        if "=" in s:
            key, value = s.split("=")
            # How to determine if value is a int, float, or str?
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            extra_config[key] = value

    # todo: parse other kwargs, if needed at all

    return arch, extra_config


def setup_model_args_tabular(args):
    # parse model name, get archictecture and extra config
    args.arch, extra_config = parse_model_name(args.model)

    if args.arch in ["mlp", "resmlp"]:
        if args.dataset.startswith("census"):
            args.in_dim = 12
            args.out_dim = 2
            args.task = "classification"
        elif args.dataset.startswith("commercial"):
            args.in_dim = 10
            args.out_dim = 2
            args.task = "classification"
        elif args.dataset.startswith("yeast"):
            args.in_dim = 8
            args.out_dim = 10
            args.task = "classification"
        elif args.dataset.startswith("wine"):
            args.in_dim = 11
            args.out_dim = 7
            args.task = "classification"
        elif args.dataset.startswith("glass"):
            args.in_dim = 9
            args.out_dim = 6
            args.task = "classification"
        elif args.dataset.startswith("telescope"):
            args.in_dim = 10
            args.out_dim = 2
            args.task = "classification"
        elif args.dataset.startswith("tictactoe"):
            args.in_dim = 9
            args.out_dim = 2
            args.task = "classification"
        elif args.dataset.startswith("raisin"):
            args.in_dim = 7
            args.out_dim = 2
            args.task = "classification"
        elif args.dataset.startswith("phishing-binary"): # todo: 注意原来是下划线_, 现在改成了-，可能会报错
            args.in_dim = 9
            args.out_dim = 2
            args.task = "classification"
        elif args.dataset.startswith("wifi"):
            args.in_dim = 7
            args.out_dim = 4
            args.task = "classification"
        else:
            raise NotImplementedError(f"Undefined Dataset: {args.dataset}")

        args.width = extra_config["width"]
        args.n_layer = extra_config["nlayer"]
        args.activation = extra_config["act"]
        args.model_kwargs = {"in_dim": args.in_dim,
                             "width": args.width,
                             "out_dim": args.out_dim,
                             "n_layer": args.n_layer,
                             "activation": args.activation}

    # todo: add other models for tabular dataset
    else:
        raise NotImplementedError(f"Undefined Model: {args.arch}")


def setup_model_args_rule(args):
    # parse model name, get archictecture and extra config
    args.arch, extra_config = parse_model_name(args.model)

    if args.arch in ["mlp", "resmlp"]:

        # todo: 整理一下这里的dataset name

        if re.match(r"gaussian_rule_(.+)_regression_10d_(.+)", args.dataset):
            args.in_dim = 10
            args.out_dim = 1
            args.task = "regression"
        elif re.match(r"binary_rule_(.+)_regression_10d_(.+)", args.dataset):
            args.in_dim = 10
            args.out_dim = 1
            args.task = "regression"
        elif re.match(r"zero_one_rule_(.+)_regression_10d_(.+)", args.dataset):
            args.in_dim = 10
            args.out_dim = 1
            args.task = "regression"
        elif re.match(r"uniform_2_rule_(.+)_regression_10d_(.+)", args.dataset):
            args.in_dim = 10
            args.out_dim = 1
            args.task = "regression"
        elif re.match(r"binary_rule_(.+)_classification_10d_(.+)", args.dataset):
            args.in_dim = 10
            args.out_dim = 2
            args.task = "classification"
        elif re.match(r"uniform_1_rule_(.+)_classification_10d_(.+)", args.dataset):
            args.in_dim = 10
            args.out_dim = 2
            args.task = "classification"
        elif re.match(r"gaussian_rule_(.+)_classification_8d_(.+)", args.dataset):
            args.in_dim = 8
            args.out_dim = 2
            args.task = "classification"
        elif re.match(r"census_rule_(.+)_regression_12d_(.+)", args.dataset):
            args.in_dim = 12
            args.out_dim = 1
            args.task = "regression"
        else:
            raise NotImplementedError(f"Undefined Dataset: {args.dataset}")

        args.width = extra_config["width"]
        args.n_layer = extra_config["n_layer"]
        args.model_kwargs = {"in_dim": args.in_dim,
                             "width": args.width,
                             "out_dim": args.out_dim,
                             "n_layer": args.n_layer}

    else:
        raise NotImplementedError(f"Undefined Model: {args.arch}")


def setup_model_args_nlp(args):
    # parse model name, get archictecture and extra config
    # Example model names: "Bert-tiny#", "Bert-medium#", "OPT-1.3b#pretrain", "llama-7b#pretrain"
    args.arch, extra_config = parse_model_name(args.model)

    # we need to specify the following 4 attributes for the model
    args.model_kwargs = {}
    args.tokenizer_kwargs = {}
    args.ModelClass = None
    args.TokenizerClass = None
    args.config_path = None

    # First, specify model_kwargs and ModelClass, based on both the dataset and the arch
    # todo: add more models and datasets, determine the model_kwargs, ModelClass, TokenizerClass, and task
    # classification tasks
    if any(args.dataset.startswith(d) for d in ["SST-2", "CoLA", "AGNews", "MNLI", "custom-imdb"]):
        if args.dataset.startswith("SST-2") and args.arch.startswith("Bert"):
            args.model_kwargs = {"num_labels": 2}
            args.ModelClass = BertForSequenceClassification
            args.TokenizerClass = AutoTokenizer
            args.task = "nlp-seq-cls" # todo: 这里的task要不要换个名字 —— 可以之后遇到要训练的情况再考虑
        elif args.dataset.startswith("CoLA") and args.arch.startswith("Bert"):
            args.model_kwargs = {"num_labels": 2}
            args.ModelClass = BertForSequenceClassification
            args.TokenizerClass = AutoTokenizer
            args.task = "nlp-seq-cls"
        elif args.dataset.startswith("AGNews") and args.arch.startswith("Bert"):
            args.model_kwargs = {"num_labels": 4}
            args.ModelClass = BertForSequenceClassification
            args.TokenizerClass = AutoTokenizer
            args.task = "nlp-seq-cls"
        elif args.dataset.startswith("MNLI") and args.arch.startswith("Bert"):
            args.model_kwargs = {"num_labels": 3}
            args.ModelClass = BertForSequenceClassification
            args.TokenizerClass = AutoTokenizer
            args.task = "nlp-nli"

        elif args.dataset.startswith("custom-imdb") and args.arch.startswith("BERTweet"): # use upper case to avoid conflicting with Bert
            args.model_kwargs = {}  # num_labels is already fixed to 3 in the BERTweet config
            args.ModelClass = AutoModelForSequenceClassification
            args.TokenizerClass = AutoTokenizer
            args.task = "nlp-seq-cls"
        else:
            raise NotImplementedError(f"Undefined arch {args.arch} or Dataset {args.dataset} for classification tasks")

    # open-ended generation, no specific model_kwargs, mostly use off-the-shelf pretrained models
    elif any(args.dataset.startswith(d) for d in ["SQuAD", "custom-squad", "custom-generation", "custom-cn-us"]):
        args.task = "nlp-generation"

        if args.arch.startswith("OPT"):
            args.model_kwargs = {}
            args.ModelClass = OPTForCausalLM
            args.TokenizerClass = AutoTokenizer
        elif args.arch.startswith("llama"):
            args.model_kwargs = {}
            args.ModelClass = LlamaForCausalLM
            args.TokenizerClass = AutoTokenizer
        elif args.arch.startswith("pythia"):
            # Here we are able to specify the revision (checkpoint) for the pythia model
            if "revision" in extra_config:
                print(f"=== Using revision {extra_config['revision']} for Pythia model ===")
                args.revision = extra_config["revision"]
                args.model_kwargs = {"revision": args.revision}
            else:
                args.model_kwargs = {}
            args.ModelClass = GPTNeoXForCausalLM
            args.TokenizerClass = AutoTokenizer
        elif args.arch.startswith("qwen2.5"):
            args.model_kwargs = {}
            args.ModelClass = AutoModelForCausalLM
            args.TokenizerClass = Qwen2TokenizerModified  # in the modified tokenizer, we can add a space at the beginning of the sentence
        else:
            raise NotImplementedError(f"Undefined arch: {args.arch}")
    else:
        raise NotImplementedError(f"Undefined Dataset: {args.dataset}")


    ### Second, specify tokenizer_kwargs and config_path, simply based on the arch
    # todo: add more model config path and tokenizer kwargs
    # Bert family
    if args.arch == "Bert-tiny":
        args.config_path = "src/models/nlp/configs/bert-tiny_config.json"
        args.tokenizer_kwargs = {}
    elif args.arch == "Bert-medium":
        args.config_path = "src/models/nlp/configs/bert-medium_config.json"
        args.tokenizer_kwargs = {}
    elif args.arch == "Bert-base":
        args.config_path = "src/models/nlp/configs/bert-base_config.json"
        args.tokenizer_kwargs = {}
    elif args.arch == "Bert-large":
        args.config_path = "src/models/nlp/configs/bert-large_config.json"
        args.tokenizer_kwargs = {}
    elif args.arch == "BERTweet":
        args.config_path = "src/models/nlp/configs/bertweet_config.json"
        args.tokenizer_kwargs = {}

    elif args.arch == "OPT-1.3b":
        args.config_path = "src/models/nlp/configs/opt-1.3b_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
        # [Important] all models using GPT2Tokenizer need this, in order to deal with different behaviors on the first word and other words
    elif args.arch == "llama-7b":
        args.config_path = "src/models/nlp/configs/llama-7b_config.json"
        args.tokenizer_kwargs = {}

    # Pythia family
    elif args.arch == "pythia-14m":
        args.config_path = "src/models/nlp/configs/pythia-14m_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "pythia-70m":
        args.config_path = "src/models/nlp/configs/pythia-70m_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "pythia-160m":
        args.config_path = "src/models/nlp/configs/pythia-160m_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "pythia-410m":
        args.config_path = "src/models/nlp/configs/pythia-410m_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "pythia-1b":
        args.config_path = "src/models/nlp/configs/pythia-1b_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "pythia-1.4b":
        args.config_path = "src/models/nlp/configs/pythia-1.4b_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "pythia-2.8b":
        args.config_path = "src/models/nlp/configs/pythia-2.8b_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "pythia-6.9b":
        args.config_path = "src/models/nlp/configs/pythia-6.9b_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "pythia-12b":
        args.config_path = "src/models/nlp/configs/pythia-12b_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}

    elif args.arch == "pythia-70m-deduped":
        args.config_path = "src/models/nlp/configs/pythia-70m-deduped_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "pythia-160m-deduped":
        args.config_path = "src/models/nlp/configs/pythia-160m-deduped_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "pythia-410m-deduped":
        args.config_path = "src/models/nlp/configs/pythia-410m-deduped_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "pythia-1b-deduped":
        args.config_path = "src/models/nlp/configs/pythia-1b-deduped_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "pythia-1.4b-deduped":
        args.config_path = "src/models/nlp/configs/pythia-1.4b-deduped_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "pythia-2.8b-deduped":
        args.config_path = "src/models/nlp/configs/pythia-2.8b-deduped_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "pythia-6.9b-deduped":
        args.config_path = "src/models/nlp/configs/pythia-6.9b-deduped_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "pythia-12b-deduped":
        args.config_path = "src/models/nlp/configs/pythia-12b-deduped_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}

    # Qwen2.5 family
    elif args.arch == "qwen2.5-0.5b":
        args.config_path = "src/models/nlp/configs/qwen2.5-0.5b_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True} # todo: 再斟酌一下，到底要不要在句子前面加空格
    elif args.arch == "qwen2.5-1.5b":
        args.config_path = "src/models/nlp/configs/qwen2.5-1.5b_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "qwen2.5-3b":
        args.config_path = "src/models/nlp/configs/qwen2.5-3b_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "qwen2.5-7b":
        args.config_path = "src/models/nlp/configs/qwen2.5-7b_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "qwen2.5-14b":
        args.config_path = "src/models/nlp/configs/qwen2.5-14b_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "qwen2.5-32b":
        args.config_path = "src/models/nlp/configs/qwen2.5-32b_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    elif args.arch == "qwen2.5-72b":
        args.config_path = "src/models/nlp/configs/qwen2.5-72b_config.json"
        args.tokenizer_kwargs = {"add_prefix_space": True}
    else:
        raise NotImplementedError(f"Undefined model: {args.arch}")

    if args.ModelClass is None or args.config_path is None:
        raise ValueError(f"ModelClass or config_path is not specified for {args.arch} on {args.dataset}")



def setup_model_args_image(args):
    # parse model name, get archictecture and extra config
    args.arch, extra_config = parse_model_name(args.model)

    if args.dataset.startswith("mnist") and args.arch in ["lenet", "resnet20", "resnet32", "resnet44", "vgg11",
                                                   "vgg13"]:
        args.model_kwargs = {"input_channel": 1, "num_classes": 10}
        args.task = "classification"
    elif args.dataset.startswith("cifar10") and args.arch in ["lenet", "resnet20", "resnet32", "resnet44",
                                                     "vgg11", "vgg13", "vgg16", "vgg19"]:
        args.model_kwargs = {"input_channel": 3, "num_classes": 10}
        args.task = "classification"
    elif args.dataset.startswith("tiny_imagenet200") and args.arch in ["alexnet", "resnet18", "resnet34", "resnet101",
                                                              "vgg13", "vgg16", "vgg11", "vgg19"]:
        args.model_kwargs = {"num_classes": 200}
        args.task = "classification"
    elif args.dataset.startswith("tiny_imagenet50") and args.arch in ["alexnet", "resnet18", "resnet34", "resnet101",
                                                             "vgg13", "vgg16", "vgg11", "vgg19"]:
        args.model_kwargs = {"num_classes": 50}
        args.task = "classification"
    elif args.dataset.startswith("cub2011") and args.arch in ["test_model", "lenet", "alexnet", "resnet20", "resnet18",
                                                     "resnet34", "resnet50", "resnet101", "vgg11", "vgg13", "vgg16",
                                                     "vgg19"]:
        args.model_kwargs = {"num_classes": 200}
        args.task = "classification"
    elif args.dataset.startswith("simpleisthree") and args.arch in ["lenet", "resnet20", "resnet32", "resnet44"]:
        args.model_kwargs = {"input_channel": 1, "num_classes": 1}
        args.task = "logistic_regression"
    elif args.dataset.startswith("celeba") and args.arch in ["alexnet", "resnet18",
                                                             "resnet34", "resnet50"]:
        args.model_kwargs = {"num_classes": 1}
        args.task = "logistic_regression"
    elif args.dataset in ["dog_bird", "reddog_bluebird", "bg_bird", "redbg_bluebird"] \
            and args.arch in ["alexnet", "resnet18", "resnet34", "resnet50"]:
        args.model_kwargs = {"num_classes": 1}
        args.task = "logistic_regression"
    else:
        raise NotImplementedError(f"[Undefined] Dataset: {args.dataset}, Model: {args.arch}")


def setup_model_args_pointcloud(args):
    # parse model name, get archictecture and extra config
    args.arch, extra_config = parse_model_name(args.model)

    if args.dataset.startswith("shapenet") and args.arch in ["pointnet", "pointnet2", "pointconv"]:
        args.model_kwargs = {"num_classes": 16}
        args.task = "pointcloud_classification"
    else:
        raise NotImplementedError(f"[Undefined] Dataset: {args.dataset}, Model: {args.arch}")