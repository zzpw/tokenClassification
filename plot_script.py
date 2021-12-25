# -*- coding: utf-8 -*-
# @Author : zengpengwu
# @Time : 2021/12/15 19:45
# @File : plot_script.py

import torch
import torch.nn.functional as F
from utils import SCORER, calculate_all_metrics
from evaluate import cal_all_metrics_on_dataset, extract_concepts_from_sentences
from dataset import ExtDataModule
from transformers import BertTokenizer, BertConfig
import matplotlib.pyplot as plt
from models import BertTokenClassifier

config = BertConfig.from_pretrained('bert-large-uncased')
model = BertTokenClassifier.from_pretrained('bert-large-uncased', config=config)
model_path = './checkpoints/2021-12-24/model.pt'
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to(torch.device(0))
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
valid_dataset = ExtDataModule(json_data_path='./ext_train.json', tokenizer=tokenizer, max_length=24)

batch_size = 64
max_length = 24

model.eval()
sentences = valid_dataset.data['sentences']
batch_num = len(sentences) // batch_size
preds = torch.Tensor().to(model.device)
for i in range(batch_num):
  _, cur = extract_concepts_from_sentences(sentences[i * batch_size: (i + 1) * batch_size], model, tokenizer,
                                           max_length=max_length)
  preds = torch.concat((preds, cur), dim=0)
  # torch.cuda.empty_cache()
_, cur = extract_concepts_from_sentences(sentences[batch_num * batch_size:], model, tokenizer, max_length=max_length)
preds = torch.concat((preds, cur))

threshold = 1.

pr_curve = dict()
for k in SCORER.keys():
  pr_curve[k] = []

while threshold >= 0.:
  # print('{:.2f}'.format(threshold))
  metrics = calculate_all_metrics(preds.detach().cpu(), valid_dataset.labels, threshold)
  for k in pr_curve.keys():
    pr_curve[k].append(metrics[k])
  threshold -= 0.01

plt.title(model_path)
plt.xlabel(xlabel='recall')
plt.ylabel(ylabel='precision')
plt.plot(pr_curve['recall'], pr_curve['precision'], label='PR-Curve')
plt.plot([0, 1], [0, 1], label='y = x', linestyle='--')
plt.legend()
plt.grid(linestyle='-.')
plt.show()

prob_rate = F.softmax(preds, dim=-1)[:, :, -1]
flatted_rate = prob_rate[valid_dataset.labels == 1]
flatted_rate = flatted_rate.detach().numpy()
plt.title('PDF')
plt.xlabel('probability')
plt.ylabel('numbers')
plt.hist(flatted_rate, bins=100, color='green')
plt.show()
