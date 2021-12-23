# -*- coding: utf-8 -*-
# @Author : zengpengwu
# @Time : 2021/12/12 15:07
# @File : dataset.py

import json

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, PreTrainedTokenizer

from utils import (
  pad_and_truncate,
  shift_data_right,
  convert_sentences_with_valid_tag,
)


class ExtDataModule(Dataset):
  def __init__(self, json_data_path, tokenizer, max_length=None):
    """
    用于加载concepts抽取的数据, 注意json数据的格式为:
     {
       'sentences': [s_1, s_2, ...],
       'labels': [l_1, l_2, ...],
       'token_idx_list': [t_1, t_2, ...],
     }

    Args:
      json_data_path (str): json格式数据的路径
      tokenizer (PreTrainedTokenizer): transformers.
      max_length (int): 句子的最大长度
    """
    super(ExtDataModule, self).__init__()
    self.json_data_path = json_data_path
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.setup_dataset()
    self.cal_valid_tag()

  def __getitem__(self, index):
    return (
      self.input_ids[index],
      self.valid_tag[index],
      self.attention_mask[index],
      self.labels[index] if self.labels is not None else None,
    )

  def __len__(self):
    return len(self.data['sentences'])

  def cal_valid_tag(self):
    self.valid_tag = []
    sentences = self.data['sentences']
    self.valid_tag = convert_sentences_with_valid_tag(sentences, self.tokenizer, self.max_length)
    self.valid_tag = torch.tensor(self.valid_tag)


  def setup_dataset(self):
    """
    set up the dataset.

    Returns:

    """
    with open(self.json_data_path, 'r') as f:
      self.data = json.load(f)
    if self.max_length is None:
      inter_data = self.tokenizer(self.data['sentences'], padding=True, return_tensors='pt')
    else:
      inter_data = self.tokenizer(self.data['sentences'], padding='max_length', truncation=True, max_length=self.max_length,
                             return_tensors='pt')
    self.input_ids = inter_data['input_ids']
    self.attention_mask = inter_data['attention_mask']
    if 'labels' in self.data:
      self.labels = pad_and_truncate(self.data['labels'], 0, self.max_length)
      for idx in range(len(self.labels)):
        self.labels[idx] = shift_data_right(self.labels[idx], 0)
      self.labels = torch.tensor(self.labels)
    else:
      self.labels = None


if __name__ == '__main__':
  tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lowercase=True)
  data = ExtDataModule('./ext_train.json', tokenizer, 24)
  dataloader = DataLoader(dataset=data, batch_size=64, shuffle=True)
