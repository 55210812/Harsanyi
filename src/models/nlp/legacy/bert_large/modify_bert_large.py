from transformers import BertForMaskedLM
from typing import List, Optional, Tuple, Union
import torch

class ModifiedBert_large(BertForMaskedLM):
    """
    Modfied model for the bert-large-acsed
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
                    return_dict=True,
                )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        return prediction_scores[:, -2, :]


if __name__ == '__main__':
    pass