#coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        """
        模型的抽象类
        """
        super(Model, self).__init__()

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



    
