import os

import numpy as np
import torch
from typing import Type
from utils.global_const import *

# Set the cache directory for huggingface hub. [Important!] It should be before the import of transformers
os.environ["HF_HOME"] = HF_HOME

from transformers import PreTrainedTokenizer


def load_baseline_embeds(
        embeds_path: str = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
    embeds_path = os.path.join(embeds_path)
    baseline_embeds_list = np.load(embeds_path)
    baseline_embeds = torch.from_numpy(baseline_embeds_list[-1]).to(dtype=dtype)
    return baseline_embeds


def get_baseline_id_nlp(
        baseline_type: str,
        tokenizer: Type[PreTrainedTokenizer],
    ) -> int:
    if baseline_type == "unk":
        baseline_id = tokenizer.unk_token_id
    elif baseline_type == "pad":
        # 注意llama的情况
        baseline_id = tokenizer.pad_token_id
    elif baseline_type == "bos":
        baseline_id = tokenizer.bos_token_id
    elif baseline_type == "eos":
        baseline_id = tokenizer.eos_token_id
    elif baseline_type == "mask":
        baseline_id = tokenizer.mask_token_id
    else:
        raise ValueError(f"baseline_type {baseline_type} not recognized.")
    return baseline_id




if __name__ == '__main__':
    # test get_baseline_id_nlp()
    from transformers import BertTokenizer, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
    tokenizer_bert = BertTokenizer.from_pretrained(
        '/data/renqihan/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'
    )
    tokenizer_opt = AutoTokenizer.from_pretrained(
        "/data/zhangjp/prove_symbolic-LLM/checkpoints/models--facebook--opt-1.3b/snapshots/8c7b10754972749675d22364c25c428b29face51"
    )
    tokenizer_llama = LlamaTokenizer.from_pretrained(
        "huggyllama/llama-7b"
    )
    tokenizer_gpt_neo = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer_llama2 = LlamaTokenizer.from_pretrained(
        "/data/zhangjp/save/llama2/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93"
    )

    llama_model = LlamaForCausalLM.from_pretrained("huggyllama/llama-7b")

    baseline_id_bert_unk = get_baseline_id_nlp("unk", tokenizer_bert)
    baseline_id_bert_pad = get_baseline_id_nlp("pad", tokenizer_bert)
    baseline_id_bert_bos = get_baseline_id_nlp("bos", tokenizer_bert)
    baseline_id_bert_eos = get_baseline_id_nlp("eos", tokenizer_bert)
    baseline_id_bert_mask = get_baseline_id_nlp("mask", tokenizer_bert)

    baseline_id_opt_unk = get_baseline_id_nlp("unk", tokenizer_opt)
    baseline_id_opt_pad = get_baseline_id_nlp("pad", tokenizer_opt)
    baseline_id_opt_bos = get_baseline_id_nlp("bos", tokenizer_opt)
    baseline_id_opt_eos = get_baseline_id_nlp("eos", tokenizer_opt)

    baseline_id_llama_unk = get_baseline_id_nlp("unk", tokenizer_llama)
    baseline_id_llama_pad = get_baseline_id_nlp("pad", tokenizer_llama)
    baseline_id_llama_bos = get_baseline_id_nlp("bos", tokenizer_llama)
    baseline_id_llama_eos = get_baseline_id_nlp("eos", tokenizer_llama)
    # todo: 我们之前下载的llama的tokenizer config好像有点问题，把unk_token, bos_token, eos_token都设为了''，但默认设置应该是bos_token='<s>', eos_token='</s'，unk_token='<unk>'(llama2从官方下载的就正常)

    baseline_id_gpt_neo_unk = get_baseline_id_nlp("unk", tokenizer_gpt_neo)
    baseline_id_gpt_neo_pad = get_baseline_id_nlp("pad", tokenizer_gpt_neo)
    baseline_id_gpt_neo_bos = get_baseline_id_nlp("bos", tokenizer_gpt_neo)
    baseline_id_gpt_neo_eos = get_baseline_id_nlp("eos", tokenizer_gpt_neo)

    baseline_id_llama2_unk = get_baseline_id_nlp("unk", tokenizer_llama2)
    baseline_id_llama2_pad = get_baseline_id_nlp("pad", tokenizer_llama2)
    baseline_id_llama2_bos = get_baseline_id_nlp("bos", tokenizer_llama2)
    baseline_id_llama2_eos = get_baseline_id_nlp("eos", tokenizer_llama2)

    print("done")