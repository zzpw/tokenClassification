# -*- coding: utf-8 -*-
# @Author : zengpengwu
# @Time : 2021/12/12 15:00
# @File : main.py

import os
import argparse
import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig
from transformers import get_linear_schedule_with_warmup
from models import BertTokenClassifier

from dataset import ExtDataModule
from evaluate import model_validation_loss, cal_all_metrics_on_dataset
from train import (
  model_forward_batch,
  set_weight_decay,
)

today = str(datetime.date.today())
# today = '2021-12-15'
parser = argparse.ArgumentParser()
parser.add_argument('--learning-rate', default=1e-5, help='learning rate of training process', type=float)
parser.add_argument('--base-model', default='bert-large-uncased', help='model backbone', type=str)
parser.add_argument('--max-length', default=24, help='maximum length of sentences', type=int)
parser.add_argument('--model-save-path', default='./checkpoints/' + today + '/model.pt', help='model save path', type=str)
parser.add_argument('--log-dir', default='./run/' + today + '/', help='tensorboard log-dir', type=str)
parser.add_argument('--epochs', default=10, help='epochs of training', type=int)
parser.add_argument('--max-grad-norm', default=1., help='clip gradients', type=float)
parser.add_argument('--cuda-device', default=[0, 1], help='available cuda device list', type=list)
parser.add_argument('--weight-decay-rate', default=1., help='l_2 regularization rate', type=float)
parser.add_argument('--batch-size', default=96, help='training batch size', type=int)
parser.add_argument('--warm-up-ratio', default=0.05, help='learning rate warm up ratio', type=float)
parser.add_argument('--label-smoothing', default=0.0, help='label smoothing regularization', type=float)
args = parser.parse_args()

model_save_path_dir = os.path.join(*os.path.split(args.model_save_path)[:-1])
if not os.path.exists(model_save_path_dir):
  os.makedirs(model_save_path_dir)

if not os.path.exists(args.log_dir):
  os.makedirs(args.log_dir)

print('-' * 80)
for k, v in vars(args).items():
  print(k, '=', v)
print('-' * 80)

config = BertConfig.from_pretrained(args.base_model)
tokenizer = BertTokenizer.from_pretrained(args.base_model)
config.label_smoothing = args.label_smoothing
model = BertTokenClassifier.from_pretrained(args.base_model, config=config)
model.to(args.cuda_device[0])
train_dataset = ExtDataModule(json_data_path='./ext_train.json', tokenizer=tokenizer, max_length=args.max_length)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_dataset = ExtDataModule(json_data_path='./ext_valid.json', tokenizer=tokenizer, max_length=args.max_length)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)

total_train_steps = args.epochs * len(train_dataloader)
warmup_steps = int(args.warm_up_ratio * total_train_steps)

optimizer = optim.AdamW(params=set_weight_decay(model, args.weight_decay_rate), lr=args.learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)

prev_valid_loss = 1000.
cur_training_step = 0
writer = SummaryWriter(args.log_dir)

for epoch in range(args.epochs):
  torch.cuda.empty_cache()
  model.train()
  for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
    loss = model_forward_batch(model, batch)
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_grad_norm)
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    cur_training_step += 1
    writer.add_scalar('Train/Loss', loss.item(), cur_training_step)
    writer.add_scalar('Train/Learning rate', scheduler.get_last_lr()[0], cur_training_step)
  torch.cuda.empty_cache()
  valid_loss = model_validation_loss(valid_dataloader=valid_dataloader, model=model)
  print('validation loss:', valid_loss)
  metrics = cal_all_metrics_on_dataset(model, tokenizer, valid_dataset)
  print(metrics)
  if valid_loss < prev_valid_loss:
    prev_valid_loss = valid_loss
    torch.save(model.state_dict(), args.model_save_path)

