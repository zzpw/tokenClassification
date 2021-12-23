# -*- coding: utf-8 -*-
# @Author : zengpengwu
# @Time : 2021/12/12 15:45
# @File : preprocess.py

from datasets import load_dataset
import nltk
import json
import nltk.stem as ns
from tqdm import tqdm
from transformers import BertTokenizer

def sent_tokenize(sent):
  """

  Args:
    sent (str): 'I am fine, thank you.'

  Returns: ['I', 'am', 'fine', ',', 'thank', 'you', '.']

  """
  return nltk.word_tokenize(sent)


def sent_lemmatize_and_tokenize(sentence):
  """

  Args:
    sentence: 'dogs and cats are playing near the tree'

  Returns: ['dog', 'and', 'cat', 'be', 'play', 'near', 'the', 'tree']

  """
  if isinstance(sentence, (list, tuple)):
    sentence = ' '.join(sentence)
  sentence = sentence.lower().strip()
  wnl = ns.WordNetLemmatizer()
  sentence_split = []
  for word, tag in nltk.pos_tag(nltk.word_tokenize(sentence)):
    if tag.startswith('NN'):
      t = wnl.lemmatize(word, pos='n')
    elif tag.startswith('VB'):
      t = wnl.lemmatize(word, pos='v')
    elif tag.startswith('JJ'):
      t = wnl.lemmatize(word, pos='a')
    elif tag.startswith('R'):
      t = wnl.lemmatize(word, pos='r')
    else:
      t = word
    t = wnl.lemmatize(t, pos='v')
    sentence_split.append(t)
  return sentence_split

def token_idx_in_sentence(sent, token):
  """

  Args:
    sent: 'how are you'
    token:  'be'

  Returns: 1

  """
  sent1 = sent_lemmatize_and_tokenize(sent)
  token1 = sent_lemmatize_and_tokenize([token])[0]
  try:
    p = sent1.index(token1)
  except ValueError:
    p = -1
  return p

if __name__ == '__main__':

  data = load_dataset('common_gen')
  train_data = data['train'].remove_columns('concept_set_idx').to_dict()
  val_data = data['validation'].remove_columns('concept_set_idx').to_dict()
  test_data = data['test'].remove_columns('concept_set_idx').to_dict()
  tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lowercase=True)

  token_idx_list = []
  sentences = []
  for concepts, target in tqdm(zip(val_data['concepts'], val_data['target']), total=len(val_data['concepts'])):
    idx_list = []
    for c in concepts:
      idx_list.append(token_idx_in_sentence(target, c))
    if not -1 in idx_list:
      token_idx_list.append(idx_list)
      sentences.append(target)

  labels = []
  for idx_list, sent in tqdm(zip(token_idx_list, sentences), total=len(token_idx_list)):
    label = [0, ] * len(nltk.word_tokenize(sent))
    for idx in idx_list:
      label[idx] = 1
    labels.append(label)

  json_obj = {'sentences': sentences, 'labels': labels, 'token_idx_list': token_idx_list}

  # with open('ext_valid.json', 'w') as f:
  #   f.write(json.dumps(json_obj))
