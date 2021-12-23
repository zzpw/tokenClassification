# -*- coding: utf-8 -*-
# @Author : zengpengwu
# @Time : 2021/12/17 16:50
# @File : common_gen_ext.py
import json

import torch
from transformers import BertTokenizer, BertConfig

from dataset import ExtDataModule
from evaluate import extract_concepts_from_sentences
from models import BertTokenClassifier
from preprocess import sent_lemmatize_and_tokenize

config = BertConfig.from_pretrained('bert-large-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertTokenClassifier.from_pretrained('bert-large-uncased')
model.load_state_dict(torch.load('./checkpoints/2021-12-16/model.pt'))
model.to(torch.device(0))
model.eval()

# %%

# train_dataset = ExtDataModule('./corpus_data.json', tokenizer, 24)
with open('./corpus_data.json', 'r') as f:
  train_data = json.load(f)
batch_size = 64
max_length = 24
sentences = train_data['sentences']
batch_num = len(sentences) // batch_size

concepts = []
t= ['A finger obstructs a currency bill that is from the US.']
extract_concepts_from_sentences(t, model, tokenizer, 24)[0]
for i in range(batch_num):
  concept, _ = extract_concepts_from_sentences(sentences[i * batch_size: (i + 1) * batch_size], model, tokenizer,
                                           max_length=max_length)
  concepts.extend(concept)
  # torch.cuda.empty_cache()
concept, _ = extract_concepts_from_sentences(sentences[batch_num * batch_size:], model, tokenizer, max_length=max_length)
concepts.extend(concept)

concepts_lemma = []
for concept in concepts:
  t = []
  for c in concept:
    t.append(sent_lemmatize_and_tokenize(c)[0])
  concepts_lemma.append(t)

additional_train_data = {'concepts': concepts_lemma, 'target': sentences}

# with open('./additional_train_data.json', 'w') as f:
#   f.write(json.dumps(additional_train_data))

# %%

