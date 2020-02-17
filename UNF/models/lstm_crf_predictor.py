#coding:utf-8
import os

import torch
from torch import nn
import torch.nn.functional as F

from models.predictor import Predictor
from models.lstm_crf import LstmCrfTagger

class LstmCrfPredictor(Predictor):
    def __init__(self, model_save_path, device=None):
        super(LstmCrfPredictor, self).__init__(model_save_path, device)

    def model_loader(self, conf):
        model = LstmCrfTagger(**conf.__dict__)
        return model

    def predict(self, input, **kwargs):
        input = input.split()
        input_ids = [self.vocab.get(item, 0) for item in input]
        input_ids = torch.LongTensor(input_ids)
        mask = torch.ones(1, input_ids.size(0))
        input_seq_length = torch.tensor([input_ids.size(0)]).long()
        if self.device is not None:
            input_ids = input_ids.to(self.device)
            mask = mask.to(self.device)

        res = self.model.predict(input_ids, input_seq_length, 
                mask)

        t_res = []
        for item in res.detach().cpu().tolist()[0]:
            t_res.append(self.target[item])
        
        return t_res
    

