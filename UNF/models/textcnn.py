#coding:utf-8
from __future__ import absolute_import
import os
import sys
print(sys.path.append(".."))
print(__name__)
import torch
from torch import nn
import torch.nn.functional as F

from models.model import Model
from modules.embedding.embedding import TokenEmbedding
from modules.encoder.cnn_maxpool import CnnMaxpoolLayer

class TextCnn(Model):
    """
    ref:Convolutional Neural Networks for Sentence Classification
    """
    def __init__(self,
                input_dim,
                vocab_size,
                filter_size,
                filter_num,
                label_nums, 
                dropout, **kwargs):
        super(TextCnn, self).__init__(input_dim, vocab_size, **kwargs)
        if not isinstance(filter_size, (tuple, list)):
            filter_size = [filter_size]

        if not isinstance(filter_num, (tuple, list)):
            filter_num = len(filter_size) * [filter_num]

        self.encoder = CnnMaxpoolLayer(input_dim,
                    filter_num, filter_size, **kwargs)
        
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(sum(filter_num), label_nums)

    def forward(self, input, label=None, mask=None):
        input = torch.transpose(input, 0, 1)
        x = self.embedding(input)
        output = self.encoder(x, mask) #[b * l]
        output = self.dropout(output)
        logits = self.fc(output) #[b, label_num]
        return {"logits": logits}




