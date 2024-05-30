from transformers import XLMRobertaModel
import torch.nn as nn

class DPRModel(nn.Module):
    def __init__(self, model_name):
        super(DPRModel, self).__init__()
        self.query_encoder = XLMRobertaModel.from_pretrained(model_name)
        self.passage_encoder = XLMRobertaModel.from_pretrained(model_name)

    def encode_queries(self, input_ids, attention_mask):
        outputs = self.query_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state.mean(dim=1)

    def encode_passages(self, input_ids, attention_mask):
        outputs = self.passage_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state.mean(dim=1)
