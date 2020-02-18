#coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F

from modules.embedding.embedding import TokenEmbedding


class Model(nn.Module):
    def __init__(self, input_dim, vocab_size, **kwargs):
        """
        模型的抽象类
        """
        super(Model, self).__init__()
        self.embedding = TokenEmbedding(input_dim, vocab_size)
        #加载预训练的词向量
        if "pretrain" in kwargs:
            if kwargs["pretrain"]:
                self.embedding.from_pretrained(kwargs['vectors'])

    def forward(self, *arg, **kwarg):
        """
        模型前向过程
        """
        raise Exception("Not implemented!!")

    def predict(self):
        """
        模型预测过程
        """
        raise Exception("Not implemented!!")

    def get_parameter_names(self):
        return [name for name, _ in self.named_parameters()]



    
