import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.electra.modeling_electra import (ElectraPreTrainedModel,
                                                          ElectraModel)

def batched_index_select(input, dim, index):
	views = [input.shape[0]] + \
		[1 if i != dim else -1 for i in range(1, len(input.shape))]
	expanse = list(input.shape)
	expanse[0] = -1
	expanse[dim] = -1
	index = index.view(views).expand(expanse)
	return torch.gather(input, dim, index)

class ElectraForEntitySpanQA(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.electra = ElectraModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.scorer = nn.Linear(config.hidden_size * 2, 1)
        # self.scorer = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size, 1)
        # )
    
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        entity_position_ids=None,
        entity_attention_mask=None,
    ):
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        entity_position_ids = entity_position_ids.clamp(min=0)
        start_entity_embeddings = batched_index_select(sequence_output, 1, entity_position_ids[:,:,0])
        end_entity_embeddings = batched_index_select(sequence_output, 1, entity_position_ids[:,:,1])
        entity_embeddings = (start_entity_embeddings + end_entity_embeddings) / 2
        entity_embedding_mask = entity_attention_mask.type_as(entity_embeddings).unsqueeze(-1)
        entity_embeddings = entity_embeddings * entity_embedding_mask

        # Seems bad but no choice
        """
        entity_embeddings = []
        for i in range(len(sequence_output)):
            entity_embeddings.append(F.embedding(entity_position_ids[i].clamp(min=0), sequence_output[i]))
        entity_embeddings = torch.stack(entity_embeddings, dim=0)
        entity_embedding_mask = (entity_position_ids != -1).type_as(entity_embeddings).unsqueeze(-1)
        entity_embeddings = entity_embeddings * entity_embedding_mask
        entity_embeddings = torch.sum(entity_embeddings, dim=-2)
        entity_embeddings = entity_embeddings / entity_embedding_mask.sum(dim=-2).clamp(min=1e-7)
        """

        doc_entity_emb = entity_embeddings[:, 1:, :]
        placeholder_emb = entity_embeddings[:, :1, :]
        feature_vector = torch.cat([placeholder_emb.expand_as(doc_entity_emb), doc_entity_emb], dim=2)
        feature_vector = self.dropout(feature_vector)
        logits = self.scorer(feature_vector)

        # feature_vector = self.scorer(entity_embeddings)
        # doc_entity_emb = feature_vector[:, 1:, :]
        # placeholder_emb = feature_vector[:, :1, :]
        # logits = torch.matmul(doc_entity_emb, placeholder_emb.transpose(1, 2))

        doc_entity_mask = entity_attention_mask[:, 1:]
        if labels is None:
            return logits.squeeze(-1) + ((doc_entity_mask - 1) * 10000).type_as(logits)

        loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits), reduction='mean')
        loss = loss.masked_select(doc_entity_mask.reshape(-1).bool()).sum()
        loss = loss / doc_entity_mask.sum().type_as(loss)

        return (loss, )