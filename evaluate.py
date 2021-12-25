# -*- coding: utf-8 -*-
# @Author : zengpengwu
# @Time : 2021/12/13 22:29
# @File : evaluate.py

from typing import List, Union, Tuple

import nltk
import torch
from dataset import ExtDataModule
from utils import convert_sentences_with_valid_tag, calculate_all_metrics
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from train import model_forward_batch


@torch.no_grad()
def model_validation_loss(valid_dataloader, model) -> float:
  """
  Calculate loss on validation dataset.

  Args:
    valid_dataloader (DataLoader):
    model (PreTrainedModel):

  Returns:

  """
  model.eval()
  total_loss = 0
  for batch in valid_dataloader:
    loss = model_forward_batch(model, batch)
    total_loss += loss.item()
  batch_loss = total_loss / len(valid_dataloader)
  return batch_loss


@torch.no_grad()
def extract_concepts_from_sentences(sentences, model, tokenizer, max_length=None):
  """
  Extract concepts from given sentences.

  Args:
    sentences (List[str]):
    model (PreTrainedModel):
    tokenizer (PreTrainedTokenizer):
    max_length (int):

  Returns: Extracted Concepts and Logits before softmax.

  """
  model.eval()
  input_ids, attention_mask, valid_tag = convert_sentences_with_valid_tag(sentences, tokenizer, max_length)
  input_ids = torch.tensor(input_ids).to(model.device)
  attention_mask = torch.tensor(attention_mask).to(model.device)
  valid_tag = torch.tensor(valid_tag).to(model.device)
  # if max_length is None:
  #   outputs = model(**tokenizer(sentences, padding=True, return_tensors='pt', ).to(model.device), valid_ids=valid_tag)
  # else:
  #   outputs = model(**tokenizer(sentences, padding='max_length', truncation=True, return_tensors='pt', max_length=max_length).to(model.device), valid_ids=valid_tag)
  outputs = model(input_ids=input_ids, attention_mask=attention_mask, valid_ids=valid_tag)
  logits = outputs.logits  # [batch_size, src_length, num_labels]
  preds = torch.argmax(logits, dim=-1)  # [batch_size, src_length]
  tokenized_sents = [nltk.word_tokenize(x) for x in sentences] # [batch_size, scr_length]
  concepts = []
  for sent, pred in zip(tokenized_sents, preds.cpu().detach().numpy()):
    c = [sent[idx - 1] for idx, i in enumerate(pred) if i]
    concepts.append(c)
  return concepts, logits

@torch.no_grad()
def cal_all_metrics_on_dataset(model, tokenizer, dataset, batch_size=64, max_length=24, threshold=0.5):
  """
  Calculate all the metrics on given dataset.

  Metrics contain: Accuracy, Precision, Recall, F1

  Args:
    model (PreTrainedModel):
    tokenizer (PreTrainedTokenizer):
    dataset (Dataset):
    batch_size (int):
    max_length (int):

  Returns:

  """
  model.eval()
  sentences = dataset.data['sentences']
  batch_num = len(sentences) // batch_size
  preds = torch.Tensor().to(model.device)
  for i in range(batch_num):
    _, cur = extract_concepts_from_sentences(sentences[i * batch_size: (i + 1) * batch_size], model, tokenizer,
                                                   max_length=max_length)
    preds = torch.concat((preds, cur), dim=0)
    torch.cuda.empty_cache()
  _, cur = extract_concepts_from_sentences(sentences[batch_num * batch_size:], model, tokenizer, max_length=max_length)
  preds = torch.concat((preds, cur))
  metrics = calculate_all_metrics(preds.detach().cpu(), dataset.labels, threshold)
  return metrics

from transformers import BertTokenizer, BertConfig
from models import BertTokenClassifier


if __name__ == '__main__':
  import json
  config = BertConfig.from_pretrained('bert-large-uncased')
  model = BertTokenClassifier.from_pretrained('bert-large-uncased', config=config)
  model.load_state_dict(torch.load('./checkpoints/2021-12-25/model.pt'))
  model.to(torch.device(1))
  tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
  valid_dataset = ExtDataModule('./ext_valid.json', tokenizer=tokenizer, max_length=24)
  metrics = cal_all_metrics_on_dataset(model, tokenizer, valid_dataset)


  # real_concepts = []
  # for sent, label in zip(sentences, valid_dataset.data['labels']):
  #   tokenized_sent = nltk.word_tokenize()