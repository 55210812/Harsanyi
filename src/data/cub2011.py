import numpy as np
import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torchvision.utils import save_image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from copy import deepcopy
from tqdm import tqdm

class Cub2011Dataset(Dataset):
    """
    references:https://github.com/TDeVries/cub2011_dataset
    """
    base_folder = 'CUB_200_2011/crop_images'
    argument_folder = "CUB_200_2011/argument_images"
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True, argument_save=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.argument_save = argument_save

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.argument_save:
            save_path = os.path.join(self.root, self.argument_folder, sample.filepath)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image(img, save_path)

        return img, target


class Cub2011(object):
    """
    reference:https://gitlab.com/artelabsuper/engraf-net/-/blob/master/dataset_load.py?ref_type=heads
    """
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
            train_transform_list.append(transforms.RandomResizedCrop(size=(224, 224), antialias=True)) # scale=(0.8, 1) why?

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

        self.trainset = Cub2011Dataset(root=self.data_root, train=True, download=True, transform=train_transforms)
        self.testset = Cub2011Dataset(root=self.data_root, train=False, download=True, transform=test_transforms)
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

        self.modified_indices = []
        for idx in indices_to_randomize:
            if self.add_noise_label == 1:
                # Randomly choose a label between 0 and 200 (inclusive)
                random_label = np.random.RandomState(idx).randint(0, 200)
            # todo: add more noise label options
            else:
                raise NotImplementedError

            sample = self.trainset.data.iloc[idx]
            if random_label != sample['target'] - 1: # target starts from 1
                new_series = pd.Series({'img_id':sample['img_id'],
                                        'filepath':sample['filepath'],
                                        'target':random_label + 1,
                                        'is_training_img':sample['is_training_img']})
                self.trainset.data.iloc[idx] = new_series # not able to directly modify the target field, need to replace the whole series
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


def bounding_boxes_image(rootpath):
    import os
    import pandas as pd
    from PIL import Image
    from shutil import copyfile

    def makedir(path):
        '''
        if path does not exist in the file system, create it
        '''
        if not os.path.exists(path):
            os.makedirs(path)

    # set paths
    imgspath = rootpath + 'crop_images/'
    savepath = rootpath + 'crop_images/'
    makedir(savepath)
    # read img names, bounding_boxes
    names = pd.read_table(rootpath + 'images.txt', delimiter=' ', names=['id', 'name'])
    names = names.to_numpy()
    boxs = pd.read_table(rootpath + 'bounding_boxes.txt', delimiter=' ',
                        names=['id', 'x', 'y', 'width', 'height'])
    boxs = boxs.to_numpy()

    # crop imgs
    for i in range(11788):
        im = Image.open(imgspath + names[i][1])
        im = im.crop((boxs[i][1], boxs[i][2], boxs[i][1] + boxs[i][3], boxs[i][2] + boxs[i][4]))
        im.save(savepath + names[i][1], quality=95)
        print('{} imgs cropped and saved.'.format(i + 1))
    print('All Done.')

    # # mkdir for cropped imgs
    # folders = pd.read_table(rootpath + 'classes.txt', delimiter=' ', names=['id', 'folder'])
    # folders = folders.to_numpy()
    # for i in range(200):
    #     makedir(trainpath + folders[i][1])
    #     makedir(testpath + folders[i][1])

    # # split imgs
    # labels = pd.read_table(rootpath + 'train_test_split.txt', delimiter=' ', names=['id', 'label'])
    # labels = labels.to_numpy()
    # for i in range(11788):
    #     if(labels[i][1] == 1):
    #         copyfile(imgspath + names[i][1], trainpath + names[i][1])
    #     else:
    #         copyfile(imgspath + names[i][1], testpath + names[i][1])
    #     print('{} imgs splited.'.format(i + 1))
    # print('All Done.')

