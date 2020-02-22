#coding:utf-8
import os
import sys
import argparse
import json

import torch
from torch import nn

from models.textcnn import TextCnnTrace
from models.fasttext import FastTextTrace
from models.dpcnn import DpCnnTrace
from models.model_util import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                         type=str,
                         required=True,
                        )
    parser.add_argument("--model_cls",
                         type=str,
                         default="TextCnnTrace",
                        )  
    parser.add_argument("--save_path",
                         type=str,
                         default="trace.pt",
                        )                      
    args = parser.parse_args()
    model_path = args.model_path
    model_cls = args.model_cls
    save_path = args.save_path

    config = Config.from_json_file("%s/conf.json" % (model_path))
    net = globals()[model_cls](**config.__dict__)
    net.load_state_dict_trace(torch.load("%s/best.th" % model_path))
    net.eval()

    mock_input = net.mock_input_data()
    tr = torch.jit.trace(net, mock_input)
    print(tr.code)

    #move vocab
    os.system("mv %s/vocab.txt trace/" % model_path)

    #save trace model
    tr.save("trace/%s" % save_path)
