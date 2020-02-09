#coding:utf-8
"""
从配置文件反射拿到learner
"""
from .learner import Trainer

class LearnerLoader(object):

    @classmethod
    def from_params(cls, mode, train_iter, dev_iter, learner_conf):
        return Trainer(model, train_iter, dev_iter,
                **learner_conf)

    
