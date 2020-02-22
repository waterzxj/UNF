#coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F

from modules.embedding.embedding import TokenEmbedding


class Model(nn.Module):
    def __init__(self):
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

    def load_state_dict(self, state_dict, strict=True):
        true_state_dict = {}
        for k,v in state_dict.items():
            if k.startswith("model."):
                k = k.split(".", 1)[1] #去掉名字中的第一个model.
            true_state_dict[k] = v

        self.model.load_state_dict(true_state_dict, strict)



    
