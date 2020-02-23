#coding:utf-8
"""
Embedding类的抽象
"""
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

from modules.module_util import init_tensor
from modules.base_type import InitType, FAN_MODE, ActivationType


class BaseEmbedding(nn.Module):
    """
    Emebdding类的基类
    :params dim int类型，embedding的维度大小
        :params vocab_size int类型
        :params device string or [string1, string2],计算的后端，默认是cpu
        :params init_type string, 初始化的计算方式 ，默认采用uniform初始化
        :params dropout float
    """

    def __init__(self, dim, vocab_size,
                device=None, dropout=0.0):

        super(BaseEmbedding, self).__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.device = device
        self.dropout = nn.Dropout(p=dropout)

    

    @classmethod
    def from_dict(cls, params):
        return cls(**params)

    def forward(self, input):
        raise Exception("BaseEmbedding forward method not implemented!")


class TokenEmbedding(BaseEmbedding):
    def __init__(self, dim, vocab_size, device=None,
                    dropout=0.0, 
                    init_type=InitType.XAVIER_NORMAL,
                    low=0, high=1, mean=0, std=1,
                    activation_type=ActivationType.NONE,
                    fan_mode=FAN_MODE.FAN_IN, negative_slope=0
                    ):
        """
        Embedding类的基础类

        :params dim int类型，embedding的维度大小
        :params vocab_size int类型
        :params device string or [string1, string2],计算的后端，默认是cpu
        :params init_type string, 初始化的计算方式 ，默认采用uniform初始化
        :params dropout float
        """
        super(TokenEmbedding, self).__init__(dim, vocab_size, device,
                                                dropout)

        self.embeddings = nn.Embedding(vocab_size, dim)
        embedding_lookup_table = init_tensor(tensor=torch.empty(vocab_size, dim),
                init_type=init_type, low=low, high=high, mean=mean, std=std,
                activation_type=activation_type, fan_mode=fan_mode, 
                negative_slope=negative_slope)

        self.embeddings.weight.data.copy_(embedding_lookup_table)
    
    def forward(self, input):
        embedding = self.embeddings(input)
        return self.dropout(embedding)

    @classmethod
    def from_pretrained(cls, vectors, vocab_map=None):
        """
        copy从dataloader每个域加载好的预训练的词向量

        :params vectors Vector类型
        """
        if isinstance(path, (str)):
            raise Exception("Load embedding from path not implemented!")
        
        self.embeddings.weight.data.copy_(vectors)


        


