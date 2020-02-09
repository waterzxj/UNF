#coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F

from .model import Model

class FastText(Model):
    """
    
    """
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.hidden_fc = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.final_fc  = nn.Linear(config.hidden_dim, config.num_labels)

    def forward(self, *arg, **kwarg):
        embedding_o = self.embedding(arg[0])
        hidden_fc_o = self.hidden_fc(embedding_o)
        final_fc_o  = self.final_fc(hidden_fc_o)

        return final_fc_o
    