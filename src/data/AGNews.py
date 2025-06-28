import os

from utils.global_const import *

# Set the cache directory for huggingface hub. [Important!] It should be before the import of transformers
os.environ["HF_HOME"] = HF_HOME

from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import json
from datasets import load_dataset


class AGNews(object):
    dataset_name_hf = "fancyzhx/ag_news"

    def __init__(self,
                 mode,
                 tokenizer,
                 batch_size=1,
                 data_path=None,
                 padding=False,
                 max_length=None): # optional data path to the pre-downloaded dataset (in case huggingface is unreachable)
        assert mode in ['train', 'eval']
        self.mode = mode
        self.batch_size = batch_size if self.mode == 'train' else 1  # override the batch_size to 1 if mode is eval
        self.padding = padding if self.mode == 'train' else False
        self.max_length = max_length if self.mode == 'train' else None
        self.tokenizer = tokenizer
        self.data_path = data_path  # a path like ".../huggingface/datasets/fancyzhx___ag_news"

        self.is_data_loaded = False

    def load_data(self):
        if self.data_path is not None:
            dataset_name = self.data_path
        else:
            dataset_name = self.dataset_name_hf

        self.train_set_raw = load_dataset(dataset_name, split='train')
        self.test_set_raw = load_dataset(dataset_name, split='test')

        self.train_set = self.train_set_raw.map(lambda item:
                                                self.tokenizer(item["text"],
                                                               return_tensors='pt',
                                                               padding=self.padding,
                                                               max_length=self.max_length,
                                                               truncation=True),
                                                batched=True,
                                                batch_size=self.batch_size)

        self.test_set = self.test_set_raw.map(lambda item:
                                              self.tokenizer(item["text"],
                                                             return_tensors='pt',
                                                             padding=self.padding,
                                                             max_length=self.max_length,
                                                             truncation=True),
                                              batched=True,
                                              batch_size=self.batch_size)

        self.train_set.set_format("torch") # transform all elements to torch tensosrs
        self.test_set.set_format("torch")
        self.is_data_loaded = True


    def get_dataloader(self, shuffle_train=False):
        if not self.is_data_loaded:
            self.load_data()

        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=shuffle_train,
                                  num_workers=4, drop_last=True)
        test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return train_loader, test_loader


if __name__ == '__main__':
    # test AGNews_hf

    tokenizer = BertTokenizer.from_pretrained(
        '/data/renqihan/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594')

    dataset = AGNews(batch_size=128, tokenizer=tokenizer, mode='eval')
    train_loader, test_loader = dataset.get_dataloader()
