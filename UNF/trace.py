#coding:utf-8
import os
import sys

import torch
from torch import nn

from models.textcnn import TextCnn
from models.model_util import Config

config = Config.from_json_file("imdb_textcnn2/conf.json")

net = TextCnn(**config.__dict__)
net.load_state_dict(torch.load("imdb_textcnn2/best.th"))
net.eval()
tr = torch.jit.trace(net, (torch.ones((1, 8), dtype=torch.long), torch.ones((1, 8), dtype=torch.long)))
print(tr.code)

tr.save('text_cnn_trace1.pt')
