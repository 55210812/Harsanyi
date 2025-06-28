import os
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import BertTokenizer


class MNLI(object):
    def __init__(self, data_root, dataset_name='MNLI', tokenizer=None, use_ratio=1.0, max_length=128):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self._is_data_loaded = False
        self.tokenizer = tokenizer
        self.use_ratio = use_ratio
        self.max_length = max_length

    def _load_data(self):
        save_path = os.path.join(self.data_root, self.dataset_name, f'Bert-use_ratio{self.use_ratio}')
        if os.path.exists(os.path.join(save_path, "train_set_input_ids.pt")):
            print("Loading data from saved files...")
            self.train_input_ids = torch.load(os.path.join(save_path, 'train_set_input_ids.pt'))
            self.train_token_type_ids = torch.load(os.path.join(save_path, 'train_set_token_type_ids.pt'))
            self.train_attention_mask = torch.load(os.path.join(save_path, 'train_set_attention_mask.pt'))
            self.train_labels = torch.load(os.path.join(save_path, 'train_set_target.pt'))

            self.test_input_ids = torch.load(os.path.join(save_path, 'test_set_input_ids.pt'))
            self.test_token_type_ids = torch.load(os.path.join(save_path, 'test_set_token_type_ids.pt'))
            self.test_attention_mask = torch.load(os.path.join(save_path, 'test_set_attention_mask.pt'))
            self.test_labels = torch.load(os.path.join(save_path, 'test_set_target.pt'))
        else:
            print("Loading data from Parquet files...")
            data_folder = os.path.join(self.data_root, self.dataset_name)
            train_data_file = os.path.join(data_folder, 'train-00000-of-00001.parquet')
            test_data_file = os.path.join(data_folder, 'validation_matched-00000-of-00001.parquet')

            # Load data from Parquet files
            train_data = pd.read_parquet(train_data_file)
            test_data = pd.read_parquet(test_data_file)

            self.train_len = len(train_data)
            self.test_len = len(test_data)

            self.train_premise = train_data['premise'].tolist()[:int(self.train_len * self.use_ratio)]
            self.train_hypothesis = train_data['hypothesis'].tolist()[:int(self.train_len * self.use_ratio)]
            self.train_labels = train_data['label'].tolist()[:int(self.train_len * self.use_ratio)] # int, not string
            self.test_premise = test_data['premise'].tolist()
            self.test_hypothesis = test_data['hypothesis'].tolist()
            self.test_labels = test_data['label'].tolist()

            print("Tokenizing data...")
            # Tokenize data
            self.train_data_dict = self.tokenizer(self.train_premise, self.train_hypothesis,  # ask the tokenizer to tokenize a pair of sentences
                                             return_tensors='pt',
                                             padding='max_length',
                                             max_length=self.max_length,
                                             truncation='longest_first')

            self.test_data_dict = self.tokenizer(self.test_premise, self.test_hypothesis,
                                             return_tensors='pt',
                                             padding='max_length',
                                             max_length=self.max_length,
                                             truncation='longest_first')

            self.train_input_ids = self.train_data_dict['input_ids']
            self.train_token_type_ids = self.train_data_dict['token_type_ids']
            self.train_attention_mask = self.train_data_dict['attention_mask']
            self.test_input_ids = self.test_data_dict['input_ids']
            self.test_token_type_ids = self.test_data_dict['token_type_ids']
            self.test_attention_mask = self.test_data_dict['attention_mask']

            os.makedirs(save_path, exist_ok=True)
            torch.save(self.train_input_ids, os.path.join(save_path, 'train_set_input_ids.pt'))
            torch.save(self.train_token_type_ids, os.path.join(save_path, 'train_set_token_type_ids.pt'))
            torch.save(self.train_attention_mask, os.path.join(save_path, 'train_set_attention_mask.pt'))
            torch.save(self.train_labels, os.path.join(save_path, 'train_set_target.pt'))

            torch.save(self.test_input_ids, os.path.join(save_path, 'test_set_input_ids.pt'))
            torch.save(self.test_token_type_ids, os.path.join(save_path, 'test_set_token_type_ids.pt'))
            torch.save(self.test_attention_mask, os.path.join(save_path, 'test_set_attention_mask.pt'))
            torch.save(self.test_labels, os.path.join(save_path, 'test_set_target.pt'))

        self._is_data_loaded = True


    def get_dataloader(self, batch_size, shuffle_train=True):
        if not self._is_data_loaded:
            self._load_data()
        train_dataset = TensorDataset(self.train_input_ids,
                                      self.train_token_type_ids,
                                      self.train_attention_mask,
                                      torch.tensor(self.train_labels))
        test_dataset = TensorDataset(self.test_input_ids,
                                     self.test_token_type_ids,
                                     self.test_attention_mask,
                                     torch.tensor(self.test_labels))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

