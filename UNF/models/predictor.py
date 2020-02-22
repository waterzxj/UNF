#coding:utf-8
import os 
import json

import torch
from torch import nn
import torch.nn.functional as F

from models.model_util import Config
from models.dpcnn import DpCnn
from models.fasttext import FastText
from models.leam import LEAM
from models.self_attention import SelfAttention
from models.textcnn import TextCnn


class Predictor(nn.Module):
    def __init__(self, model_save_path, device=None, model_type=None):
        super(Predictor, self).__init__()
        model_conf = os.path.join(model_save_path, "conf.json")
        vocab_path = os.path.join(model_save_path, "vocab.txt")
        target_path = os.path.join(model_save_path, "target.txt")

        self.model_type = model_type
        self.model = self.model_loader(Config.from_json_file(model_conf))
        self.model.load_state_dict(torch.load(os.path.join(model_save_path, "best.th")))
        self.model.eval()
                                    
        self.device = device
        if self.device is not None:
            self.model.to(device)

        self.vocab = self.load_vocab(vocab_path)
        self.target = self.load_vocab(target_path, reverse=True)

    def model_loader(self, conf):
        name = self.model_type.lower()
        if name == "textcnn":
            model = TextCnn(**conf.__dict__)
        elif name == "fastext":
            model = FastText(**conf.__dict__)
        elif name == "dpcnn":
            model = DpCnn(**conf.__dict__)
        elif name == "leam":
            model = LEAM(**conf.__dict__)
        elif name == "self-attention":
            model = SelfAttention(**conf.__dict__)
        else:
            raise Exception("name:%s model not implemented!" % (name))

        return model

    def predict(self, input, **kwargs):
        input = input.split()
        input_ids = [self.vocab.get(item, 0) for item in input]

        input_ids = torch.LongTensor(input_ids)
        if self.device is not None:
            input_ids = input_ids.to(self.device)

        mask = (input_ids != 1).long()

        res = self.model.predict(input_ids, mask)
        res = res.detach().cpu().tolist()[0]
        return res

    def load_vocab(self, path, reverse=False):
        res = {}
        tmp = json.load(open(path))
        for index, word in enumerate(tmp):
            if reverse:
                res[index] = word
            else:
                res[word] = index
        return res
