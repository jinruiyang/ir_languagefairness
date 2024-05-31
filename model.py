from transformers import BertModel
import torch.nn as nn

# class DPRModel(nn.Module):
#     def __init__(self, model_name):
#         super(DPRModel, self).__init__()
#         self.query_encoder = BertModel.from_pretrained(model_name)
#         self.passage_encoder = BertModel.from_pretrained(model_name)
#
#     def encode_queries(self, input_ids, attention_mask):
#         outputs = self.query_encoder(input_ids=input_ids, attention_mask=attention_mask)
#         return outputs.last_hidden_state.mean(dim=1)
#
#     def encode_passages(self, input_ids, attention_mask):
#         outputs = self.passage_encoder(input_ids=input_ids, attention_mask=attention_mask)
#         return outputs.last_hidden_state.mean(dim=1)


class DPRModel(nn.Module):
    def __init__(self, model_name):
        super(DPRModel, self).__init__()
        self.query_encoder = BertModel.from_pretrained(model_name)
        self.passage_encoder = BertModel.from_pretrained(model_name)

    def encode_queries(self, input_ids, attention_mask):
        outputs = self.query_encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = attention_mask[..., None].float()  # Convert to float for correct masking
        last_hidden_state = last_hidden_state * attention_mask  # Apply attention mask
        mean_embeddings = last_hidden_state.sum(dim=1) / attention_mask.sum(dim=1)
        return mean_embeddings

    def encode_passages(self, input_ids, attention_mask):
        outputs = self.passage_encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = attention_mask[..., None].float()  # Convert to float for correct masking
        last_hidden_state = last_hidden_state * attention_mask  # Apply attention mask
        mean_embeddings = last_hidden_state.sum(dim=1) / attention_mask.sum(dim=1)
        return mean_embeddings
