#coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F

from common.utils import initial_parameter


class CnnMaxpoolLayer(nn.Module):
    def __init__(self, input_num, output_num,
                filter_size, stride=1, padding=0,
                activation='relu', init_method=None)
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

        assert len(filter_size) == len(output_num), 
                "Filter size len is not equal output_num len"

        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=input_num, output_channels=on, 
                        kernal_size=ks, stride=stride, padding=padding) 
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
        conv_res = [self.activation(conv(x)) for conv in self.convs] #[b, o, lout]

        if mask is not None:
            mask = mask.unsquuze(1) # [b, 1, l]
            xs = [x.masked_fill_(mask, float('-inf')) for x in xs]

        tmp = [F.MaxPool1d(input=x, kernal_size=x.size(2)).squeeze(2) for x in xs]
        return torch.cat(tmp, dim=-1)









