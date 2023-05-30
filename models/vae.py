import sys
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist

import transformers
from transformers.activations import ACT2FN

from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, RobertaPooler


# TODO: Support general model

class Variational(nn.Module):
    def __init__(self, model, **kwargs) -> None:
        super().__init__()
        self.model = model
             
        self.classifier = RobertaClassificationHead(model.config)

    def forward(self, x):
        outputs = self.model(**x)
        z = outputs[0]
        # Average Pooling
        hs = torch.mean(z, dim=1, keepdim=True)
        # Original
        # hs = z

        logits = self.classifier(hs)
        return logits, z

class VariationalForMC(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.pooler = RobertaPooler(model.config)
        self.classifier = nn.Linear(model.config.hidden_size, 1)

    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        flat: Optional[bool] = True):
        
        if flat:
            num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

            flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
            flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
            flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
            flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
            flat_inputs_embeds = (
                inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
                if inputs_embeds is not None
                else None
            )

            outputs = self.model(
                flat_input_ids,
                position_ids=flat_position_ids,
                token_type_ids=flat_token_type_ids,
                attention_mask=flat_attention_mask,
                head_mask=head_mask,
                inputs_embeds=flat_inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            z = outputs[0]
            # hs = z
            hs = torch.mean(z, dim=1, keepdim=True)   # Average pooling
            logits = self.classifier(self.pooler(hs))
            reshaped_logits = logits.view(-1, num_choices)
        else:
            num_choices = 1
            outputs = self.model(
                input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            z = outputs[0]
            hs = torch.mean(z, dim=1, keepdim=True)
            reshaped_logits = self.classifier(self.pooler(hs))
        return reshaped_logits, z

class Generator(nn.Module):
    def __init__(self, head, config) -> None:
        super().__init__()
        self.head = head
        self.config = config
    
    def forward(self, z):
        logits = self.head(z)
        return logits

class VAE(nn.Module):
    def __init__(self, config, variational, generator) -> None:
        super().__init__()
        self.config = config
        self.variational = variational
        self.generator = generator

    def forward(self, x):
        logits_cls, z = self.variational(x)
        logits_px = self.generator(z)
        return logits_cls, logits_px, z
    
    # def kl(self, mean, logstd):
    #     # Prior: N(0, 1)
    #     return -0.5 * torch.sum(1. + logstd - mean.pow(2) - logstd.exp())