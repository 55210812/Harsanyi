from transformers import AutoTokenizer, BertGenerationDecoder, BertGenerationConfig
from typing import List, Optional, Tuple, Union
import torch

class ModifiedBert(BertGenerationDecoder):
    """
    Modfied model for the bert
    Input the input_ids or the input_embeds
    Output the logit of the predicted word
    """
    def __init__(self, config):
        super().__init__(config)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        ) -> torch.Tensor:
        
        if input_ids is not None:
            attention_mask = torch.ones_like(input_ids).to(self.device)
        if inputs_embeds is not None:
            attention_mask = torch.ones_like(inputs_embeds[:,:,0],dtype=torch.int64).to(self.device)

        outputs = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    past_key_values=None,
                    use_cache=False,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None,
                )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)[:,-1]
        return prediction_scores


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
    config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
    config.is_decoder = True
    model = ModifiedBert.from_pretrained(
        "google/bert_for_seq_generation_L-24_bbc_encoder", config=config
    )
    model.eval()

    inputs = tokenizer("Hello, my dog is cute", return_token_type_ids=False, return_tensors="pt")["input_ids"]
    outputs = model(inputs)
    print(model)

    print(outputs.shape)
    print(outputs)