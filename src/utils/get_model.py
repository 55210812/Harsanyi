import torch
import torch.nn as nn
import os
import json
import models.image as image_models
import models.tabular as tabular_models
import models.nlp as nlp_models

from utils.global_const import *

# Set the cache directory for huggingface hub. [Important!] It should be before the import of transformers
os.environ["HF_HOME"] = HF_HOME


def load_checkpoint(model, ckpt_path, device) -> None:
    """
    Load checkpoint from disk
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'File doesn\'t exists {ckpt_path}')
    print(f'=> loading checkpoint "{ckpt_path}"')
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Makes us able to load models saved with legacy versions
    if not ('state_dict' in checkpoint):
        sd = checkpoint
    else:
        sd = checkpoint['state_dict']

    # load with models trained on a single gpu or multiple gpus
    if 'module.' in list(sd.keys())[0]:
        sd = {k[len('module.'):]: v for k, v in sd.items()}

    model.load_state_dict(sd)
    print(f'=> loaded checkpoint "{ckpt_path}"')


def get_model_image(args, mode: str):
    """ get model and load parameters if needed
    :param args: arguments
    :param mode: "eval" or "train"
    :return model: model to be used
    """
    # torch.hub.set_dir(args.pretrained_models_dirname)
    assert mode in ["eval", "train"]

    # get model
    model = image_models.__dict__[args.arch](**args.model_kwargs) # note: args.arch is already parsed in setup_model_args() function

    # load checkpoint
    # If mode == 'eval':
    #   If init is in the model name, we don't load any checkpoint
    #   Note that we need to load the pretrained model if we use a pretrained model (e.g., vgg11#imgnet-pretrain),
    #    because unlike huggingface, image models are not loaded with pretrained weights by default
    # If mode == 'train':
    #   Only when pretrain is in the model name, we need to load the pretrained model (for finetuning)
    if (mode == "eval" and not "init" in args.model) or (mode == "train" and "pretrain" in args.model):
        with open(CKPT_PATH_FILE, "r") as f:
            ckpt_path_dict = json.load(f)
        args.ckpt_path = ckpt_path_dict[args.dataset][args.model]
        load_checkpoint(model, args.ckpt_path, args.device)

    model = model.to(args.device)
    if mode == "eval":
        model.eval()
    return model


def get_model_tabular(args, mode: str):
    """ get model and load parameters if needed
    :param args: arguments
    :param mode: "eval" or "train"
    :return model: model to be used
    """
    # torch.hub.set_dir(args.pretrained_models_dirname)
    assert mode in ["eval", "train"]

    # get model
    model = tabular_models.__dict__[args.arch](**args.model_kwargs) # note: args.arch is already parsed in setup_model_args() function

    # load checkpoint
    # If mode == 'eval':
    #   If init is in the model name, we don't load any checkpoint
    #   Note that we need to load the pretrained model if we use a pretrained model (e.g., vgg11#imgnet-pretrain),
    #    because unlike huggingface, image models are not loaded with pretrained weights by default
    # If mode == 'train':
    #   Only when pretrain is in the model name, we need to load the pretrained model (for finetuning)
    if (mode == "eval" and not "init" in args.model) or (mode == "train" and "pretrain" in args.model):
        with open(CKPT_PATH_FILE, "r") as f:
            ckpt_path_dict = json.load(f)
        args.ckpt_path = ckpt_path_dict[args.dataset][args.model]
        load_checkpoint(model, args.ckpt_path, args.device)

    model = model.to(args.device)
    if mode == "eval":
        model.eval()
    return model


def get_model_nlp(args, mode: str):
    """ get model and load parameters if needed
    :param args: arguments
    :param mode: "eval" or "train"
    :return model: model to be used
    """
    assert mode in ["eval", "train"]

    # Get model
    # Specify ModelClass and config_path
    # We use a Calculator wrapper to handle the model
    if any(args.dataset.startswith(d) for d in ["SST-2", "AGNews", "CoLA", "MNLI", "custom-imdb"]):

        model = nlp_models.CalculatorForSeqCls(config_path=args.config_path,
                                               ModelClass=args.ModelClass,
                                               TokenizerClass=args.TokenizerClass,
                                               model_kwargs=args.model_kwargs, # The model_kwargs may contain num_labels, etc.
                                               tokenizer_kwargs=args.tokenizer_kwargs)

    elif any(args.dataset.startswith(d) for d in ["SQuAD", "custom-squad", "custom-generation", "custom-cn-us"]):

        model = nlp_models.CalculatorForLMGeneration(config_path=args.config_path,
                                                     ModelClass=args.ModelClass,
                                                     TokenizerClass=args.TokenizerClass,
                                                     model_kwargs=args.model_kwargs, # The model_kwargs may be empty
                                                     tokenizer_kwargs=args.tokenizer_kwargs)

    else:
        raise NotImplementedError(f"Undefined dataset: {args.dataset}")


    # load checkpoint
    # If mode == 'eval':
    #   If init is in the model name, we don't load any checkpoint
    #   If we just use the pretrained model without any finetuning, we don't load any checkpoint further
    # If mode == 'train':
    #   The pretrained model is already loaded (in most cases for huggingface models), so no need to load any checkpoint here
    # todo: 下面这个load是否对LLM也适用？
    if mode == "eval" and not "init" in args.model and not "pretrain" in args.model:
        with open(CKPT_PATH_FILE, "r") as f:
            ckpt_path_dict = json.load(f)
        args.ckpt_path = ckpt_path_dict[args.dataset][args.model]
        load_checkpoint(model, args.ckpt_path, args.device)

    model = model.to(args.device)
    if mode == "eval":
        model.eval()
    return model