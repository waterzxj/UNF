#coding:utf-8
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F

from modules.module_util import initial_parameter


class CnnMaxpoolLayer(nn.Module):
    def __init__(self, input_num, output_num,
                filter_size, stride=1, padding=0,
                activation='relu', initial_method=None, **kwargs):
        """
        cnn+maxpooling的结构做encoder
        :params input_num int 输入的维度
        :params output_num int|list 每个卷积核输出的维度
        :params filter_size list 卷积核的大小
        :params init_method str 网络参数初始化的方法，默认为xavier_uniform
        """
        super(CnnMaxpoolLayer, self).__init__()
        if not isinstance(filter_size, (tuple, list)):
            filter_size = [filter_size]

        if not isinstance(output_num, (tuple, list)):
            output_num = [output_num] * len(filter_size)

        assert len(filter_size) == len(output_num), \
                "Filter size len is not equal output_num len"

        print("stride", stride)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=input_num, out_channels=on, 
                        kernel_size=ks, stride=stride, padding=padding) 
                        for on, ks in zip(output_num, filter_size)]
        )

        if activation == "relu":
            self.activation = F.relu
        elif activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        else:
            raise Exception("%s activation not support" % activation)

        #使用默认初始化
        #initial_parameter(self, initial_method)

    def forward(self, input, mask=None):
        """
        :params: input torch.Tensor [batch_size, length, dim]
        :params: mask torch.Tensor [batch_size, length]
        """
        if mask is not None:
            input = input * mask.unsqueeze(-1).float()

        #[b, l, d] -> [b, d, l]
        input = torch.transpose(input, 1, 2)
        conv_res = [self.activation(conv(input)) for conv in self.convs] #[b, o, lout]

        tmp = [F.max_pool1d(input=x, kernel_size=x.size(2)).squeeze(2) for x in conv_res]
        return torch.cat(tmp, dim=-1)









