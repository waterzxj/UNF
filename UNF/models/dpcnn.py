#coding:utf-8
from __future__ import absolute_import
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F

from model import Model


class DpCnn(Model):
    def __init__(self, input_dim, vocab_size, label_nums,
                    block_size=16, filter_size=3, filter_num=250,
                    stride=2, dropout=0.0, **kwargs):
        """
        implementation: Deep Pyramid Convolutional Neural Networks for Text Categorization

        :params block_size dpcnn里block的数量，每个block包括两个卷积层和一个max_pooling层
        """
        super(DpCnn, self).__init__(input_dim, vocab_size, **kwargs)
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.stride = stride
        self.padding = int(self.filter_size/2)
        assert self.filter_size % 2 == 1, "filter size should be odd"
        self.block_size = block_size

        self.region_embedding = torch.nn.Sequential(
                        torch.nn.Conv1d(
                        input_dim, self.filter_num,
                        self.filter_size, padding=self.padding)
        )

        self.blocks = torch.nn.ModuleList([torch.nn.Sequential(
                        torch.nn.ReLU(),
                        torch.nn.Conv1d(
                                self.filter_num, self.filter_num,
                                self.filter_size, padding=self.padding),
                        torch.nn.ReLU(),
                        torch.nn.Conv1d(
                                self.filter_num, self.filter_num,
                                self.filter_size, padding=self.padding)
        ) for _ in range(self.block_size + 1)])

        self.linear = nn.Linear(self.filter_num, label_nums)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, label=None, mask=None):
        input = self.embedding(input)

        if mask is not None:
            input = input * mask.unsqueeze(-1).float()

        #region embedding
        input = input.transpose(1, 2)
        region_out = self.region_embedding(input)
        block_out = self.blocks[0](region_out)
        #short cut
        block_out = block_out + region_out
        for index in range(1, self.block_size+1):
            block_features = F.max_pool1d(
                block_out, self.filter_size, self.stride)
            block_out = self.blocks[index](block_features)
            block_features = block_features + block_out
        doc_embedding = F.max_pool1d(
            block_features, block_features.size(2)).squeeze()

        logits = self.dropout(self.linear(doc_embedding))
        return {"logits": logits}

    def predict(self, input, label=None, mask=None):
        return self.forward(input, label, mask)["logits"]


        







