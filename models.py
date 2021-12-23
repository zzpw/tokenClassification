# -*- coding: utf-8 -*-
# @Author : zengpengwu
# @Time : 2021/12/13 16:17
# @File : models.py

# Copied and modified from transformers

import torch
import torch.utils.checkpoint
from torch import nn
from loss import FocalLoss
from transformers.models.bert.modeling_bert import (
  TokenClassifierOutput,
  BertPreTrainedModel,
  BertModel,
)


class BertTokenClassifier(BertPreTrainedModel):
  """
  Bert for token classification.

  Only the first token in sub-words counts.

  Different from vanilla bert, you should input valid_ids to point out which token matters.
  """

  _keys_to_ignore_on_load_unexpected = [r"pooler"]

  def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels

    self.bert = BertModel(config, add_pooling_layer=False)
    classifier_dropout = (
      config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
    )
    self.dropout = nn.Dropout(classifier_dropout)
    self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    if hasattr(config, 'label_smoothing'):
      self.loss_fct = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    else:
      self.loss_fct = nn.CrossEntropyLoss()
    self.loss_fct = FocalLoss(self.loss_fct)
    self.init_weights()

  def forward(
    self,
    input_ids=None,
    attention_mask=None,
    valid_ids=None,  # [batch_size, src_length]
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
  ):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
      input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )

    sequence_output = outputs[0]  # [batch_size, src_length, emd_dim]
    batch_size, src_length, embed_dim = sequence_output.size()
    # [batch_size, src_length, emd_dim]

    #  使用 valid_ids 把不属于每个 token 第一个 sub-word 的 token 屏蔽
    valid_outputs = torch.zeros([batch_size, src_length, embed_dim], dtype=torch.float32, device=self.device)
    # valid_outputs1 = torch.zeros([batch_size, src_length, embed_dim], dtype=torch.float32, device=self.device)
    for i in range(batch_size):
      cur_out = sequence_output[i][valid_ids[i] == 1]  # [length_, emd_dim]
      valid_outputs[i][:cur_out.shape[0]] = cur_out
      # p = 0
      # for j in range(src_length):
      #   if valid_ids[i][j].item():
      #     valid_outputs[i][p] = sequence_output[i][j]
      #     p += 1
    # print(torch.equal(valid_outputs, valid_outputs1))
    sequence_output = self.dropout(valid_outputs)
    logits = self.classifier(sequence_output)  # [batch_size, src_length, num_labels]

    loss = None
    if labels is not None:
      # loss_fct = CrossEntropyLoss(label_smoothing=0.1)
      # Only keep active parts of the loss
      if attention_mask is not None:
        active_loss = attention_mask.view(-1) == 1  # [batch_size * src_length, ]
        active_logits = logits.view(-1, self.num_labels)  # [batch_size * src_length, num_labels]
        #  [batch_size * src_length, ]
        active_labels = torch.where(
          active_loss, labels.view(-1), torch.tensor(self.loss_fct.ignore_index).type_as(labels)
        )
        loss = self.loss_fct(active_logits, active_labels)  # [batch_size * src_length, num_labels]  , [batch * src_length]
      else:
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # [batch_size * src_length, num_labels], [batch_size * src_length]

    if not return_dict:
      output = (logits,) + outputs[2:]
      return ((loss,) + output) if loss is not None else output

    return TokenClassifierOutput(
      loss=loss,
      logits=logits,
      hidden_states=outputs.hidden_states,
      attentions=outputs.attentions,
    )
