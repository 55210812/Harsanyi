from data import (AGNews, MNLI, MNIST, TinyImageNet50, Cub2011, CIFAR10, SST_2, CustomTextDataset)


def get_dataset_image(dataset_name, batch_size, data_root, train_augs):
    """
    Get dataset for image classification
    :param dataset_name: name of the dataset
    :param batch_size: batch size for the dataloader
    :param data_root: root directory of the dataset, necessary for image datasets
    :param train_augs: Bool, whether to apply data augmentation for training
    :return: dataset object
    """
    if dataset_name == "mnist":
        return MNIST(data_root, batch_size, train_augs=train_augs)
    elif dataset_name == "mnist-noise0.01":
        return MNIST(data_root, batch_size, train_augs=train_augs, add_noise_label=1, noise_label_ratio=0.01, seed=0)
    elif dataset_name == "tiny_imagenet50":
        return TinyImageNet50(data_root, batch_size, train_augs=train_augs)
    elif dataset_name == "cub2011":
        return Cub2011(data_root, batch_size, train_augs=train_augs)
    elif dataset_name == "cub2011-noise0.05":
        return Cub2011(data_root, batch_size, train_augs=train_augs, add_noise_label=1, noise_label_ratio=0.05, seed=0)
    elif dataset_name == "cifar10":
        return CIFAR10(data_root, batch_size, train_augs=train_augs)
    elif dataset_name == "cifar10-noise0.01":
        return CIFAR10(data_root, batch_size, train_augs=train_augs, add_noise_label=1, noise_label_ratio=0.01, seed=0)
    # todo: add more datasets
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")


def get_dataset_nlp(dataset_name, mode, tokenizer, batch_size, data_path=None, **kwargs):
    """
    Get dataset for NLP tasks
    :param dataset_name: name of the dataset
    :param mode: "train" or "eval", "eval" is used for computing interactions
    :param tokenizer: huggingface tokenizer object
    :param batch_size: batch size for the dataloader and the tokenizer
    :param data_path: [Optional] path to the pre-downloaded dataset (in case huggingface is unreachable)
    :param kwargs: additional keyword arguments to pass to the dataset object, e.g., padding, max_length...
    :return: dataset object
    """
    if dataset_name == "SST-2":
        return SST_2(mode=mode, tokenizer=tokenizer, batch_size=batch_size, data_path=data_path, **kwargs)
    elif dataset_name == "AGNews":
        return AGNews(mode=mode, tokenizer=tokenizer, batch_size=batch_size, data_path=data_path, **kwargs)
    elif dataset_name == "MNLI-0.1":
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    elif dataset_name.startswith("custom-"): # mode must be 'eval', batchsize is fixed to 1
        return CustomTextDataset(mode=mode, tokenizer=tokenizer, data_path=data_path, **kwargs)
    # todo: add more datasets
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
