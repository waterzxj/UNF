#coding:utf-8
import sys
import torch
from torch import nn
import torch.nn.functional as F

from models.model import Model

class FastText(Model):
    """
    implementation: Bag of Tricks for Efficient Text Classification
    
    """
    def __init__(self, input_dim, vocab_size,
                hidden_dim, label_nums, **kwargs):
        super(FastText, self).__init__(input_dim, vocab_size, **kwargs)
        self.hidden_fc = nn.Linear(input_dim, hidden_dim)
        self.final_fc  = nn.Linear(hidden_dim, label_nums)

    def forward(self, input, mask=None, label=None):
        embedding_o = self.embedding(input) #b * s * dim
        #embedding_o = embedding_o.permute(0,2,1) #b * dim * s

        tmp = F.avg_pool2d(embedding_o, (embedding_o.size(1), 1)).squeeze(1)#b * dim

        hidden_fc_o = self.hidden_fc(tmp)
        final_fc_o  = self.final_fc(hidden_fc_o)

        return {"logits": final_fc_o}

    def predict(self, input, mask=None, label=None):
        return self.forward(input, mask, label)["logits"]
    
