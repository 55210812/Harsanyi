import os
import os.path as osp
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.global_const import *

# Set the cache directory for huggingface hub. [Important!] It should be before the import of transformers
os.environ["HF_HOME"] = HF_HOME

from transformers import BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader


class SST_2(object):
    dataset_name_hf = "stanfordnlp/sst2"

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
        self.data_path = data_path  # a path like ".../huggingface/datasets/stanfordnlp___sst2"

        self.is_data_loaded = False

    def load_data(self):
        if self.data_path is not None:
            dataset_name = self.data_path
        else:
            dataset_name = self.dataset_name_hf

        self.train_set_raw = load_dataset(dataset_name, split='train') # ('idx', 'sentence', 'label')
        self.val_set_raw = load_dataset(dataset_name, split='validation')

        self.train_set = self.train_set_raw.map(lambda item:
                                                self.tokenizer(item["sentence"],
                                                               return_tensors='pt',
                                                               padding=self.padding,
                                                               max_length=self.max_length,
                                                               truncation=True),
                                                batched=True,
                                                batch_size=self.batch_size)

        self.val_set = self.val_set_raw.map(lambda item:
                                            self.tokenizer(item["sentence"],
                                                           return_tensors='pt',
                                                           padding=self.padding,
                                                           max_length=self.max_length,
                                                           truncation=True),
                                            batched=True,
                                            batch_size=self.batch_size)

        # ('idx', 'sentence', 'label', 'input_ids', 'attention_mask', 'token_type_ids')

        self.train_set.set_format("torch") # transform all elements to torch tensosrs
        self.val_set.set_format("torch")
        self.is_data_loaded = True


    def get_dataloader(self, shuffle_train=False):
        if not self.is_data_loaded:
            self.load_data()

        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=shuffle_train,
                                  num_workers=4, drop_last=True)
        val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader



if __name__ == '__main__':
    # test SST_2_Huggingface

    tokenizer = BertTokenizer.from_pretrained(
        '/data/renqihan/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594')
    dataset = SST_2(batch_size=128, tokenizer=tokenizer, mode='eval')
    train_loader, val_loader = dataset.get_dataloader()

    for batch in train_loader:
        print(batch)
        break


