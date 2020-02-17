#coding:utf-8
import os
import sys
import argparse
import json

from models.lstm_crf_predictor import LstmCrfPredictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                         type=str,
                         required=True,
                        )
    parser.add_argument("--device",
                         type=str,
                         default=None,
                         required=True,
                        )
    parser.add_argument("--test_path",
                         type=str,
                         required=True,
                        )  
    parser.add_argument("--save_path",
                         type=str,
                         default="test_save.dat",
                         required=True,
                        )                      

    args = parser.parse_args()
    model_path = args.model_path
    device = args.device
    #step1 初始化predictor
    predictor = LstmCrfPredictor(model_path, device)

    #step2 开始预测
    save_path = open(os.path.join(model_path, args.save_path), "w")
    for line in open(args.test_path):
        line = json.loads(line.rstrip())
        pred = predictor.predict(line["TEXT"])
        save_path.write("%s\t%s\t%s\n" % (line["TEXT"], line["LABEL"], " ".join(pred)))

    save_path.close()










