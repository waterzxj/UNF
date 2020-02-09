#coding:utf-8
"""
从配置文件反射到对应的模型
"""

from fasttext import FastText
from textcnn import TextCnn

class ModelLoader(object):

    @classmethod
    def from_params(cls, model_conf, fields):
        if len(model_conf) == 1:
            extra = {}
            name = model_conf["name"]
            #hardcode label的field_name
            if "label_num" not in model_conf:
                label_num = len(fields["LABEL"].vocab.stoi)
            else:
                label_num = model_conf["label_num"]

            vocab_size = len(fields[name].vocab.stoi)
        
            encoder_params = model_conf["encoder_params"]
            if "pretrained" in encoder_params and encoder_params["pretrained"]:
                extra["vectors"] = fields[name].vocab.vectors

      
            return globals()[encoder_cls](label_num=label_num, vocab_size=vocab_size, 
                        **model_conf["encoder_params"], **extra)
        else:
            #多域模型
            pass
