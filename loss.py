# -*- coding: utf-8 -*-
# @Author : zengpengwu
# @Time : 2021/12/17 15:53
# @File : loss.py
import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
  # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
  def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
    super(FocalLoss, self).__init__()
    self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
    self.gamma = gamma
    self.alpha = alpha
    self.reduction = loss_fcn.reduction
    self.ignore_index = loss_fcn.ignore_index
    self.loss_fcn.reduction = 'none'  # required to apply FL to each element

  def forward(self, pred, true):
    """

    Args:
      pred: [batch_size, src_length, num_labels]
      true: [batch_size, src_length]

    Returns:

    """
    loss = self.loss_fcn(pred, true)
    # p_t = torch.exp(-loss)
    # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

    # [batch_size, src_length, num_labels]
    # pred_prob = torch.sigmoid(pred)  # prob from logits
    pred_prob = F.softmax(pred, dim=-1)  # [batch_size, src_length, num_labels]
    p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
    alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
    modulating_factor = (1.0 - p_t) ** self.gamma
    loss *= alpha_factor * modulating_factor

    if self.reduction == 'mean':
      return loss.mean()
    elif self.reduction == 'sum':
      return loss.sum()
    else:  # 'none'
      return loss
