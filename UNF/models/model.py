#coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F

from .metric import F1Measure


class BaseModel(nn.Module):
    def __init__(self):
        """
        模型的抽象类
        """
        super(BaseModel, self).__init__()

    def forward(self, *arg, **kwarg):
        """
        模型前向过程
        """
        raise Exception("Not implemented!!")

    def get_metrics(self):
        """
        返回模型训练的metrics指标
        """
        raise Exception("Not implemented!!")

    def get_regularization(self):
        """
        返回模型训练时参数正则的结果，训练的时候会加入到loss里面计算
        """
        raise Exception("Not implemented!!")

    def predict(self):
        """
        模型预测过程
        """
        raise Exception("Not implemented!!")



    