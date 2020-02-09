#coding:utf-8
"""
训练一个textcnn文本分类模型的工作流
"""
from conf.textcnn_conf import data_loader_conf,model_conf,learner_conf
from data.data_loader import DataLoader
from models.model_loader import ModelLoader
from training.learner_loader import LearnerLoader

data_loader = DataLoader(data_loader_conf)
train_iter, dev_iter, test_iter = data_loader.generate_dataset()

model = ModelLoader.from_params(model_conf, data_loader.fields)
learner = LearnerLoader.from_params(model, train_iter, dev_iter, learner_conf)

learner.learn()



