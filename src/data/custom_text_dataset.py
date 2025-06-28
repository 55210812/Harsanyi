import os
import os.path as osp
import re

from utils.global_const import *

# Set the cache directory for huggingface hub. [Important!] It should be before the import of transformers
os.environ["HF_HOME"] = HF_HOME

from transformers import BertTokenizer
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

# A dataset object that loads custom texts from files
class CustomTextDataset(object):
    def __init__(self,
                 mode,
                 tokenizer,
                 data_path, # data_path must be provided for the custom text dataset
                 data_file_name="sentences.txt",
                 label_file_name="labels.txt",
                 padding=False,
                 max_length=None):
        assert mode in ['eval'], "Only 'eval' mode is supported for CustomTextDataset"
        self.mode = mode
        self.batch_size = 1  # override the batch_size to 1 if mode is eval
        self.padding = padding if self.mode == 'train' else False
        self.max_length = max_length if self.mode == 'train' else None
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.data_file_name = data_file_name
        self.label_file_name = label_file_name

        self.has_labels = False
        # check if there are label files
        if os.path.isfile(os.path.join(self.data_path, self.label_file_name)):
            self.has_labels = True

        self.is_data_loaded = False


    def load_data(self):
        # Read the text files
        with open(os.path.join(self.data_path, self.data_file_name), 'r') as f:
            sentences = f.readlines()

        # Clean up the data (remove any extra whitespace or newlines)
        sentences = [sentence.strip() for sentence in sentences]

        if self.has_labels:
            with open(os.path.join(self.data_path, self.label_file_name), 'r') as f:
                labels = f.readlines()
                labels = [int(label.strip()) for label in labels]
                data_dict = {'text': sentences, 'label': labels}
        else:
            data_dict = {'text': sentences}

        self.dataset_raw = Dataset.from_dict(data_dict)

        self.dataset = self.dataset_raw.map(lambda item:
                                            self.tokenizer(item["text"],
                                                           return_tensors='pt',
                                                           padding=self.padding,
                                                           max_length=self.max_length,
                                                           truncation=True),
                                            batched=True,
                                            batch_size=self.batch_size)

        # ('text', 'input_ids', 'attention_mask', 'token_type_ids', [optional] 'label')

        self.dataset.set_format("torch") # transform all elements to torch tensosrs
        self.is_data_loaded = True


    def get_dataloader(self, shuffle_train=False): # shuffle_train is a dummy argument
        if not self.is_data_loaded:
            self.load_data()

        train_loader = None  # Todo: For now, it is just a placeholder, we only output a test_loader
        test_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return train_loader, test_loader



if __name__ == '__main__':
    # test
    tokenizer = BertTokenizer.from_pretrained(
        '/data/renqihan/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594')
    dataset = CustomTextDataset(tokenizer=tokenizer,
                                mode='eval',
                                data_path='../datasets/custom-test-squad')
    _, test_loader = dataset.get_dataloader()

    for batch in test_loader:
        print(batch)
        break


