#coding:utf-8
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F

class FullConnectLayer(nn.Module):
    def __init__(self, in_features, out_features,
                    dropout=0.0, act="relu"):
        """
        封装torch.nn.Linear(),加入droput和activate
        """
        super(FullConnectLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act

        if self.act != None:
            if self.act == "relu":
                self.act_func = F.relu
            elif self.act == "sigmoid":
                self.act_func = F.sigmoid
            elif self.act == "tanh":
                self.act_func = F.tanh
            else:
                raise Exception("%s activation not support" % act)

    def forward(self, input):
        tmp = self.dropout(self.fc(input))
        if self.act:
            tmp = self.act_func(tmp)

        return tmp


