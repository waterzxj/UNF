#coding:utf-8
"""
从配置文件反射到对应的模型
"""
import sys

from models.fasttext import FastText
from models.textcnn import TextCnn
from models.lstm_crf import LstmCrfTagger
from models.dpcnn import DpCnn
from models.self_attention import SelfAttention
from models.leam import LEAM


class ModelLoader(object):

    @classmethod
    def from_params(cls, model_conf, fields):
        if len(model_conf) == 1:
            model_conf = model_conf[0]
            extra = {}
            name = model_conf["name"]
            #hardcode label的field_name
            if "label_num" not in model_conf:
                label_num = len(fields["LABEL"][1].vocab.stoi)
            else:
                label_num = model_conf["label_num"]

            model_conf["encoder_params"]["label_nums"] = label_num

            vocab_size = len(fields[name][1].vocab.stoi)
            model_conf["encoder_params"]["vocab_size"] = vocab_size
        
            encoder_params = model_conf["encoder_params"]
            if "pretrained" in encoder_params and encoder_params["pretrained"]:
                extra["vectors"] = fields[name].vocab.vectors

      
            return globals()[model_conf["encoder_cls"]](**model_conf["encoder_params"], **extra), \
                    model_conf["encoder_params"]
        else:
            #多域模型
            pass
