#coding:utf-8
"""
训练一个文本分类模型的工作流
加载数据 -> 加载模型 -> 训练保存

以下提供了各个分类模型的conf文件
"""
import sys
#from conf.lstm_crf_conf import data_loader_conf,model_conf,learner_conf
#from conf.dpcnn_conf import data_loader_conf,model_conf,learner_conf
#from conf.textcnn_conf import data_loader_conf,model_conf,learner_conf
from conf.fasttext_conf import data_loader_conf,model_conf,learner_conf
#from conf.leam_conf import data_loader_conf,model_conf,learner_conf
#from conf.selfattention_conf import data_loader_conf,model_conf,learner_conf
from data.data_loader import DataLoader
from models.model_loader import ModelLoader
from training.learner_loader import LearnerLoader


data_loader = DataLoader(data_loader_conf)
train_iter, dev_iter, test_iter = data_loader.generate_dataset()

model, model_conf = ModelLoader.from_params(model_conf, data_loader.fields)
learner = LearnerLoader.from_params(model, train_iter, dev_iter, learner_conf, test_iter=test_iter, fields=data_loader.fields, model_conf=model_conf)

learner.learn()



