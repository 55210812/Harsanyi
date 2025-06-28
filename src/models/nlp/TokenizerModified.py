import os
from utils.global_const import *

# Set the cache directory for huggingface hub. [Important!] It should be before the import of transformers
os.environ["HF_HOME"] = HF_HOME

from transformers import (Qwen2Tokenizer, AutoTokenizer)
import unicodedata

class Qwen2TokenizerModified(Qwen2Tokenizer):
    def __init__(self, add_prefix_space=True, **kwargs):
        super().__init__(**kwargs)
        self.add_prefix_space = add_prefix_space

    def prepare_for_tokenization(self, text, **kwargs): # override the method
        if self.add_prefix_space:
            text = " " + text
        text = unicodedata.normalize("NFC", text)
        return (text, kwargs)