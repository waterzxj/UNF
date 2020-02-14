#coding:utf-8
import os
import sys
sys.path.append("modules")
import torch
from torch import nn
import torch.nn.functional as F

from module_util import initial_parameter


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

        initial_parameter(self, initial_method)

    def forward(self, input, mask=None):
        """
        :params: input torch.Tensor [batch_size, length, dim]
        :params: mask torch.Tensor [batch_size, length]
        """
        #[b, l, d] -> [b, d, l]
        input = torch.transpose(input, 1, 2)
        conv_res = [self.activation(conv(input)) for conv in self.convs] #[b, o, lout]

        if mask is not None:
            mask = mask.unsquuze(1) # [b, 1, l]
            #mask对conv操作没有影响，但是可能对max_pool操作产生影响，在max_pool之前对mask填充-inf
            conv_res = [x.masked_fill_(mask, float('-inf')) for x in conv_res]

        tmp = [F.max_pool1d(input=x, kernel_size=x.size(2)).squeeze(2) for x in conv_res]
        return torch.cat(tmp, dim=-1)









