#coding:utf-8
"""
从配置文件反射拿到learner
"""
import os
import sys
import logging
sys.path.append("training")
from learner import Trainer

logger = logging.getLogger(__name__)

class LearnerLoader(object):

    @classmethod
    def from_params(cls, model, train_iter, dev_iter, learner_conf, test_iter=None, fields=None):
        if fields is not None:
            if "label_tag" in learner_conf:
                label_index = fields["LABEL"][1].vocab.stoi[learner_conf["label_tag"]]
                logger.info("Load label index")
                logger.info("Label index: %s" % label_index)
                logger.info("Label vocab: %s" % fields["LABEL"][1].vocab.stoi)
                return Trainer(model, train_iter, dev_iter,
                        **learner_conf, test_iter=test_iter, label_index=label_index)

        return Trainer(model, train_iter, dev_iter,
                        **learner_conf, test_iter=test_iter)

    
