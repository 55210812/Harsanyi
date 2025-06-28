import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import os
from copy import deepcopy
from tqdm import tqdm


class TinyImageNet50(object):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self,
                 data_root,
                 batch_size,
                 add_noise_label=0,
                 noise_label_ratio=0.,
                 seed=0,
                 train_augs=False,
                 ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.add_noise_label = add_noise_label
        self.noise_label_ratio = noise_label_ratio
        self.seed = seed

        self.train_augs = train_augs
        self.is_data_loaded = False


    def load_data(self):
        train_transform_list = []

        if self.train_augs:
            train_transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            train_transform_list.append(transforms.RandomResizedCrop(size=(224, 224), antialias=True))

        train_transform_list.extend([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

        train_transforms = transforms.Compose(train_transform_list)

        test_transforms = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
        ])

        self.trainset = datasets.ImageFolder(root=f'{self.data_root}/tiny-imagenet-50/train/', transform=train_transforms)
        self.testset = datasets.ImageFolder(root=f'{self.data_root}/tiny-imagenet-50/val_split/', transform=test_transforms)

        self._is_data_loaded = True


    def _add_noise_label(self):
        num_samples = len(self.trainset)
        num_samples_to_randomize = int(self.noise_label_ratio * num_samples)
        indices_to_randomize = np.random.RandomState(self.seed).choice(np.arange(num_samples),
                                                                       size=num_samples_to_randomize,
                                                                       replace=False)
        indices_to_randomize = np.sort(indices_to_randomize)
        # print("len(indices_to_randomize)", len(indices_to_randomize))
        # print("indices_to_randomize", indices_to_randomize)
        # self.ori_train_targets = deepcopy(self.trainset.targets)
        self.modified_indices = []
        for idx in tqdm(indices_to_randomize):
            if self.add_noise_label == 1:
                # Randomly choose a label between 0 and 49 (inclusive)
                random_label = np.random.RandomState(idx).randint(0, 50)
            # todo: add more noise label options
            else:
                raise NotImplementedError

            if random_label != self.trainset.targets[idx]:
                self.trainset.targets[idx] = random_label
                self.trainset.imgs[idx] = (self.trainset.imgs[idx][0], random_label) # tuple (image_path, label) - we need modify the tuple as a whole
                self.modified_indices.append(idx)


    def get_dataloader(self, shuffle_train=False):
        if not self.is_data_loaded:
            self.load_data()

        if self.add_noise_label != 0:
            self._add_noise_label()

        train_loader = DataLoader(self.trainset, batch_size=self.batch_size,
                                  shuffle=shuffle_train, num_workers=4, drop_last=True)

        # Randomly shuffle the test set (with a specified seed), in order to get a fixed yet shuffled test set
        # random_indices_testset = np.random.RandomState(self.seed).choice(np.arange(len(self.testset)),
        #                                                                  size=len(self.testset), replace=False)

        test_loader = DataLoader(self.testset, batch_size=self.batch_size,
                                 shuffle=False, num_workers=4)

        return train_loader, test_loader



