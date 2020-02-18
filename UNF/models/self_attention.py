#coding:utf-8
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

from models.model import Model
from modules.embedding.embedding import TokenEmbedding
from modules.encoder.lstm_encoder import LstmEncoderLayer
from modules.encoder.self_attention_encoder import SelfAttentionEncoder

class SelfAttention(Model):
    
    def __init__(self, label_nums, vocab_size, input_dim,  
                hidden_size, layer_num, attention_num, coefficient=0.1, bidirection=True, 
                batch_first=True, device=None,
                dropout=0.0, averge_batch_loss=True, **kwargs):
        """
        Implemention: A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING

        :params label_nums int 输出label的数量
        :params vocab_size int 词表大小
        :params input_dim int 输入词表维度
        :params hidden_size int 隐藏层的维度
        :params layer_num int 隐藏层的层数
        :params attention_num int attention的个数
        :params coefficient float 正则化系数
        """
        super(SelfAttention, self).__init__(input_dim, vocab_size, **kwargs)
        self.coefficient = coefficient
    
        self.encoder = LstmEncoderLayer(input_dim, hidden_size, num_layers, label_nums=label_nums,
                bidirectional=bidirection, batch_first=batch_first, dropout=dropout)

        self.averge_batch_loss = averge_batch_loss

        self.att_encoder = SelfAttentionEncoder(attention_num, hidden_size, hidden_size/4,
                                                coefficient )
        self.fc = nn.Linear(hidden_size * attention_num, label_nums)

    def forward(self, input, input_seq_length, label=None, mask=None):
        embedding = self.embedding(input) #batch_size * seq_len * input_dim
        encoder_res = self.encoder(embedding, input_seq_length) #batch * seq_len * (hidden_size)
        batch_size = encoder_res.size(0)
        seq_len = encoder_res.size(1)

        #attention的实现
        att_res = self.att_encoder(encoder_res, mask)
        att_encoder = att_res["encoder"]
        logits = self.fc(att_encoder)
        output = {}
        output["logits"] = logits
        
        if self.coefficient:
            output["regulariration_loss"] = att_res["regulariration_loss"]
            output["coefficient"] = coefficient

        return output
        



