import torch.nn as nn
import torch
import os
import json
from abc import ABC, abstractmethod
from typing import Optional, Type

from utils.global_const import *

# Set the cache directory for huggingface hub. [Important!] It should be before the import of transformers
os.environ["HF_HOME"] = HF_HOME

import transformers

class Calculator(nn.Module):
    '''
    class to obtain the model output logits from input embeddings or input ids
    '''
    def __init__(self,
                 config_path: str,
                 ModelClass: Type[transformers.PreTrainedModel],
                 TokenizerClass: Type[transformers.PreTrainedTokenizer],
                 model_kwargs: Optional[dict] = None,
                 tokenizer_kwargs: Optional[dict] = None
                 ) -> None:
        super(Calculator, self).__init__()
        self.config_path = config_path
        self.ModelClass = ModelClass
        self.TokenizerClass = TokenizerClass
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs

        self._load_config()
        self._load_model(self.cal_config["model_path"])
        self._load_tokenizer(self.cal_config["tokenizer_path"])
        self.word_embed = self.get_layer(self.model, self.cal_config["word_embed"])


    def _load_config(self):
        config_file = os.path.join(self.config_path)
        with open(config_file, 'r') as f:
            self.cal_config = json.load(f)


    def _load_model(self, model_path):
        if os.path.exists(model_path):
            print(f"Calculator class: Loading model from {model_path}")
            self.model = self.ModelClass.from_pretrained(model_path,
                                                         torch_dtype='auto',
                                                         **self.model_kwargs)
        else:
            print(f"Calculator class: Loading model from huggingface")
            self.model = self.ModelClass.from_pretrained(self.cal_config["model_name"],
                                                         torch_dtype='auto',
                                                         **self.model_kwargs)


    def _load_tokenizer(self, tokenizer_path):
        if os.path.exists(tokenizer_path):
            print(f"Calculator class: Loading tokenizer from {tokenizer_path}")
            self.tokenizer = self.TokenizerClass.from_pretrained(tokenizer_path,
                                                                 **self.tokenizer_kwargs)
        else:
            print(f"Calculator class: Loading tokenizer from huggingface")
            self.tokenizer = self.TokenizerClass.from_pretrained(self.cal_config["model_name"],
                                                                 **self.tokenizer_kwargs)


    def get_layer(self, 
                  model: nn.Module,
                  layer_name: str,
        ) -> nn.Module:

        layer_list = layer_name.split("/")
        prev_module = model
        for layer in layer_list:
            prev_module = prev_module._modules[layer]

        return prev_module


    def get_embeds(self,
                   input_ids: torch.Tensor
                   ):
        with torch.no_grad():
            word_embeddings = self.word_embed(input_ids)
        return word_embeddings




class CalculatorForLMGeneration(Calculator):
    """
    Calculator wrapper for LMs
    Input the input_ids or the input_embeds
    Output the logit of the predicted word
    Can get word embed given the input_ids
    """

    def __init__(self,
                 config_path: str,
                 ModelClass: Type[transformers.PreTrainedModel],
                 TokenizerClass: Type[transformers.PreTrainedTokenizer],
                 model_kwargs: Optional[dict] = None,
                 tokenizer_kwargs: Optional[dict] = None
                 ) -> None:
        """
        - config_path: the path to the config file
        - ModelClass: the model class to be used
            e.g. OPTForCausalLM, LlamaForCausalLM, T5ForConditionalGeneration
        """
        super().__init__(config_path, ModelClass, TokenizerClass, model_kwargs, tokenizer_kwargs)
        # This superclass already loaded configs, models, tokenizers


    def forward(self,
                input_ids: torch.LongTensor = None,  # shape (batch_size, seq_len)
                inputs_embeds: Optional[torch.FloatTensor] = None,  # shape (batch_size, seq_len, embed_dim)
                attention_mask: Optional[torch.LongTensor] = None,  # shape (batch_size, seq_len)
                ) -> torch.Tensor:
        # At least one of input_ids or input_embeds should be provided
        if input_ids is None and inputs_embeds is None:
            raise ValueError("At least one of input_ids or input_embeds should be provided")

        # attention_mask = torch.ones_like(input_ids).to(self.model.device) if input_ids is not None \
        #     else torch.ones_like(inputs_embeds[:, :, 0], dtype=torch.int64).to(self.model.device)
        # shape (batch_size, seq_len)

        logits = self.model(input_ids=input_ids,
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            use_cache=False, # todo: 20250118 试一下这样能不能减小显存占用 —— 可以
                            return_dict=True)["logits"]  # shape (batch_size, seq_len, vocab_size)
        # if inputs_embeds is not None:
        #     print(f"Calculator inputs_embeds dtype: {inputs_embeds.dtype}") # torch.bfloat16 for Qwen2.5
        # if attention_mask is not None:
        #     print(f"Calculator attention_mask dtype: {attention_mask.dtype}")
        # print(f"Calculator logits dtype: {logits.dtype}")  # torch.float32 for Qwen2.5, it means the model itself transforms the output to float32

        logits_next_token = logits[:, -1, :]  # get the logits for the next token, shape (batch_size, vocab_size)

        return logits_next_token



class CalculatorForSeqCls(Calculator):
    """
    Calculator wrapper for sequence classification
    Input the input_ids or the input_embeds
    Output the logit of the predicted class
    Can get word embed given the input_ids
    """

    def __init__(self,
                 config_path: str,
                 ModelClass: Type[transformers.PreTrainedModel],
                 TokenizerClass: Type[transformers.PreTrainedTokenizer],
                 model_kwargs: Optional[dict] = None,
                 tokenizer_kwargs: Optional[dict] = None
                 ) -> None:
        """
        - config_path: the path to the config file
        - ModelClass: the model class to be used
            e.g. BertForSequenceClassification, RobertaForSequenceClassification
        """
        super().__init__(config_path, ModelClass, TokenizerClass, model_kwargs, tokenizer_kwargs)
        # This superclass already loaded configs, models, tokenizers


    def forward(self,
                input_ids: torch.LongTensor = None,  # shape (batch_size, seq_len)
                inputs_embeds: Optional[torch.FloatTensor] = None,  # shape (batch_size, seq_len, embed_dim)
                attention_mask: Optional[torch.LongTensor] = None,  # shape (batch_size, seq_len)
                # token_type_ids: Optional[torch.LongTensor] = None,  # shape (batch_size, seq_len), e.g., for NLI tasks
                ) -> torch.Tensor:
        # At least one of input_ids or input_embeds should be provided
        if input_ids is None and inputs_embeds is None:
            raise ValueError("At least one of input_ids or input_embeds should be provided")

        logits = self.model(input_ids=input_ids,
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            # token_type_ids=token_type_ids,
                            return_dict=True)["logits"]  # shape (batch_size, num_labels)

        return logits