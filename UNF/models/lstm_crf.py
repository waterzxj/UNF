#coding:utf-8

import sys

import torch
from torch import nn
import torch.nn.functional as F

from model import Model
from modules.embedding.embedding import TokenEmbedding
from modules.encoder.lstm_encoder import LstmEncoderLayer
from modules.decoder.crf import CRF

class LstmCrfTagger(Model):
    def __init__(self, input_dim, vocab_size,
                hidden_size, num_layers,
                bidirection=True, batch_first=True,
                dropout=0.0, **kwargs):
        """
        ref: Neural Architectures for Named Entity Recognition
        模型结构是word_embedding + bilstm + crf

        :params 
        """
        self.embedding = TokenEmbedding(input_dim, vocab_size)
        #加载预训练的词向量
        if "pretrain" in kwargs:
            if kwargs["pretrain"]:
                self.embedding.from_pretrained(kwargs['vectors'])

        self.lstm