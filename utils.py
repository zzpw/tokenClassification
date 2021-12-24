# -*- coding: utf-8 -*-
# @Author : zengpengwu
# @Time : 2021/12/14 19:06
# @File : utils.py
from typing import Dict, Tuple

import nltk
import torch
import torch.nn.functional as F
from sklearn.metrics import (
  accuracy_score,
  precision_score,
  recall_score,
  f1_score,
)
from transformers import PreTrainedTokenizer


SCORER = {'accuracy': accuracy_score, 'precision': precision_score, 'recall': recall_score, 'f1': f1_score}


def pad_and_truncate(x, pad, max_length=None, return_mask=False):
  """
  Pad the elements of x to max_length with 'pad' symbol.

  Args:
    x:
    pad:
    max_length (int):
    return_mask (bool):

  Returns:

  """
  if max_length is None:
    max_length = max([len(t) for t in x])
  mask = []
  for i in range(len(x)):
    if len(x[i]) >= max_length:
      x[i] = x[i][:max_length]
      mask.append([1, ] * max_length)
    else:
      x[i] = x[i] + [pad for _ in range(max_length - len(x[i]))]
      t = [1, ] * len(x[i]) + [0, ] * (max_length - len(x[i]))
      mask.append(t)
  if return_mask:
    return x, mask
  else:
    return x


def shift_data_right(x, pad):
  """
  Shift data right.

  E.g: [1, 2, 3, 4] => [pad, 1, 2, 3]

  Args:
    x: torch.tensor
    pad (int):

  Returns:

  """
  x[1:] = x[:-1]
  x[0] = pad
  return x


def convert_sentences_with_valid_tag(sentences, tokenizer, max_length=None) -> Tuple:
  """

  Args:
    sentences (list[str]):
    tokenizer (PreTrainedTokenizer):
    max_length (int):

  Returns: input_ids, attention_mask, valid_tag

  """
  valid_tag = []
  input_ids = []
  for sent in sentences:
    tokenized_sent = nltk.word_tokenize(sent)
    valid_ids = [1, ]  # [CLS]
    sent_ids = [tokenizer.cls_token_id, ]  # [CLS]
    tokens = []
    for word in tokenized_sent:
      token = tokenizer.tokenize(word)
      tokens.extend(token)
      valid_ids += [1, ] + [0, ] * (len(token) - 1)
    for token in tokens:
      sent_ids.append(tokenizer.convert_tokens_to_ids(token))
    if max_length is not None:
      valid_ids = valid_ids[:max_length - 1]
      sent_ids = sent_ids[:max_length - 1]
    valid_ids.append(1)  # [EOS]
    valid_tag.append(valid_ids)
    sent_ids.append(tokenizer.sep_token_id)  # [EOS]
    input_ids.append(sent_ids)
  input_ids, attention_mask = pad_and_truncate(input_ids, tokenizer.pad_token_id, max_length, return_mask=True)
  valid_tag = pad_and_truncate(valid_tag, 0, max_length)
  return input_ids, attention_mask, valid_tag


def calculate_metrics(preds, labels, metrics, threshold=0.5) -> float:
  """
  Calculate metrics between labels and predictions

  Args:
    preds: [batch_size, src_length, num_labels]
    labels: [batch_size, src_length]
    metrics (str):
    threshold (float):

  Returns:

  """
  if metrics not in SCORER:
    raise TypeError('No such metrics like ' + metrics)
  pred_flatted = calculate_label_with_threshold(preds, threshold)
  labels_flatted = labels.view(-1)
  if metrics == 'accuracy':
    return SCORER[metrics](labels_flatted, pred_flatted)
  else:
    return SCORER[metrics](labels_flatted, pred_flatted, pos_label=1)


def calculate_all_metrics(preds, labels, threshold=0.5) -> Dict:
  """
  Calculate all metrics between labels and predictions

  Args:
    preds: [batch_size, src_length, num_labels]
    labels: [batch_size, src_length]

  Returns: None

  """
  ans = dict()
  for metrics in SCORER.keys():
    score = calculate_metrics(preds, labels, metrics, threshold)
    ans[metrics] = score
  return ans


def calculate_label_with_threshold(preds, threshold):
  """

  Args:
    preds (torch.Tensor): [batch_size, src_length, num_labels]
    threshold (float):

  Returns:

  """
  preds = F.softmax(preds, dim=-1)
  preds = preds[:, :, -1].view(-1)
  pred_labels = (preds >= threshold).long()
  return pred_labels