#coding:utf-8
import os

import torch
from torch import nn
import torch.nn.functional as F

from predictor import Predictor
from lstm_crf import LstmCrfTagger

class LstmCrfPredictor(Predictor):
    def __init__(self, model_save_path, device=None):
        super(LstmCrfPredictor, self).__init__()

    def model_loader(self, conf):
        model = LstmCrfTagger(**conf.__dict__)
        return model

    

