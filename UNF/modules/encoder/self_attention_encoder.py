#coding:utf-8
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

from modules.module_util import mask_softmax


class  SelfAttentionEncoder(nn.Module):
    def __init__(self, head_num, input_dim, 
                    attention_dim, coefficient=None):
        """
        imple: A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING

        :params head_num int attention的个数
        :params input_dim int 输入数据的维数
        :params attention_dim attention的维度大小，超参，attention的一个中间变量
        """
        super(SelfAttentionEncoder, self).__init__()
        self.att_s1 = nn.Linear(input_dim, attention_dim, bias=False)
        self.att_s2 = nn.Linear(attention_dim, head_num, bias=False)
        self.coefficient = coefficient

    def forward(self, input, mask=None):
        inner_out = self.att_s1(input) #batch_size * seq_len * attention_dim
        final_out = self.att_s2(inner_out) #batch_size * seq_len * head_num

        #if mask is not None:
        #    mask = mask.unsqueeze(2) #batch_size * seq_len * 1

        att_weight = mask_softmax(final_out, 1, mask) #batch_size * seq_len * head_num
        
        output = {}
        output["attention"] = att_weight

        #H = A*M
        H = att_weight.transpose(1,2)@input #batch_size * atten_num * input_dim
        batch_size = input.size(0)
        output["encoder"] = H.view(batch_size, -1)

        if self.coefficient:
            output["regulariration_loss"] = self.frobenius_regularization_penalty(att_weight) / batch_size

        return output

    def frobenius_regularization_penalty(self, attention):
        """
        实现论文中PENALIZATION TERM，||AAT − I||
        """
        num_timesteps = attention.size(1)
        batch_size = attention.size(0)
        #(batch_size, num_attention_heads, timesteps)
        attention_transpose = attention.transpose(1, 2)

        identity = torch.eye(num_timesteps, device=attention.device)

        #(batch_size, timesteps, timesteps)
        identity = identity.unsqueeze(0).expand(batch_size, num_timesteps, num_timesteps)

        #(batch_size, timesteps, timesteps)
        delta = attention @ attention_transpose - identity

        return torch.sum(torch.sum(torch.sum(delta ** 2, 1), 1) ** 0.5)







