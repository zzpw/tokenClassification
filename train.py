# -*- coding: utf-8 -*-
# @Author : zengpengwu
# @Time : 2021/12/13 18:09
# @File : train.py

from transformers import PreTrainedModel
from typing import Dict, List, Tuple, Any, Union

def move_batch_to_device(batch, device) -> Tuple:
  """

  Args:
    batch (Tuple, List): batch数据
    device: torch device

  Returns: tuple

  """
  batch = tuple(x.to(device) for x in batch)
  return batch


def model_forward_batch(model, batch):
  """

  Args:
    model (PreTrainedModel):
    batch (Tuple):

  Returns:

  """
  batch = move_batch_to_device(batch, model.device)
  b_input_ids, b_valid_ids, b_attention_mask, b_labels = batch
  output = model(input_ids=b_input_ids, valid_ids=b_valid_ids, attention_mask=b_attention_mask, labels=b_labels)
  loss = output.loss
  # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
  # optimizer.step()
  # model.zero_grad()
  return loss

def set_weight_decay(model, weight_decay_rate, full_fine_tune=True) -> List[Dict]:
  """

  Args:
    model (PreTrainedModel): transformers' model.
    weight_decay_rate (float): regularization rate.
    full_fine_tune (bool): fine-tune all parameters or not.

  Returns:

  """
  no_decay = ['bias', 'gamma', 'beta']

  param_optimizer = list(model.named_parameters())
  if full_fine_tune:
    # split params into two grouped: decay and no_decay.
    grouped_params = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
       'weight_decay_rate': 0.00},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
       'weight_decay_rate': weight_decay_rate}
    ]
  else:
    grouped_params = [{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                       'weight_decay_rate': 0.00}]

  return grouped_params