#coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F

from modules.embedding.embedding import TokenEmbedding


class ModelTrace(nn.Module):
    def __init__(self, input_dim=None, vocab_size=None, **kwargs):
        """
        trace类的抽象类
        """
        super(ModelTrace, self).__init__()
        if input_dim is not None and vocab_size is not None:
            self.embedding = TokenEmbedding(input_dim, vocab_size)
            #加载预训练的词向量
            if "pretrain" in kwargs:
                if kwargs["pretrain"]:
                    self.embedding.from_pretrained(kwargs['vectors'])

    def forward(self, *arg, **kwarg):
        """
        模型前向过程
        """
        raise Exception("Not implemented!!")

    def mock_input_data(self):
        """
        mock trace输入
        """
        return torch.ones((1, 128), dtype=torch.long), torch.ones((1, 128), dtype=torch.long)

    def load_state_dict_trace(self, state_dict, strict=True):
        true_state_dict = {}
        for k,v in state_dict.items():
            if k.startswith("model."):
                k = k.split(".", 1)[1] #去掉名字中的第一个model.
            true_state_dict[k] = v

        self.load_state_dict(true_state_dict, strict)

    def get_parameter_names(self):
        return [name for name, _ in self.named_parameters()]


    
