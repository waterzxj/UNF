#coding:utf-8

import torch

from learner_util import get_ner_BIO


class Metric(object):
    def __call__(self,
                 predictions,
                 gold_labels,
                 mask=None):
        """
        metric的抽象类

        :params predictions 预测结果的tensor
        :params gold_labels 实际结果的tensor
        :mask   mask
        """
        raise NotImplementedError

    def get_metric(self, reset=False):
        """
        返回metric的指标
        """
        raise NotImplementedError

    def reset(self):
        """
        重置内部状态
        """
        raise NotImplementedError

    @staticmethod
    def unwrap_to_tensors(*tensors):
        """
        把tensor安全的copy到cpu进行操作，避免gpu的oom
        """
        return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)

    @classmethod
    def from_option(cls, conf):
        return cls(**conf)


class F1Measure(Metric):
    def __init__(self, positive_label):
        """
        准确率、召回率、F值的评价指标
        """
        super(F1Measure, self).__init__()
        self._positive_label = positive_label
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0
        
    def __call__(self,
                 predictions,
                 gold_labels,
                 mask=None):
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)
        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise Exception("A gold label passed to F1Measure contains an id >= {}, "
                                     "the number of classes.".format(num_classes))
        if mask is None:
            mask = torch.ones_like(gold_labels)
        mask = mask.float()
        gold_labels = gold_labels.float()

        self.update(predictions, gold_labels, mask)

    def update(self, predictions, gold_labels, mask):
        positive_label_mask = gold_labels.eq(self._positive_label).float()
        negative_label_mask = 1.0 - positive_label_mask

        argmax_predictions = predictions.max(-1)[1].float().squeeze(-1)

        # True Negatives: correct non-positive predictions.
        correct_null_predictions = (argmax_predictions !=
                                    self._positive_label).float() * negative_label_mask
        self._true_negatives += (correct_null_predictions.float() * mask).sum()

        # True Positives: correct positively labeled predictions.
        correct_non_null_predictions = (argmax_predictions ==
                                        self._positive_label).float() * positive_label_mask
        self._true_positives += (correct_non_null_predictions * mask).sum()

        # False Negatives: incorrect negatively labeled predictions.
        incorrect_null_predictions = (argmax_predictions !=
                                      self._positive_label).float() * positive_label_mask
        self._false_negatives += (incorrect_null_predictions * mask).sum()

        # False Positives: incorrect positively labeled predictions
        incorrect_non_null_predictions = (argmax_predictions ==
                                          self._positive_label).float() * negative_label_mask
        self._false_positives += (incorrect_non_null_predictions * mask).sum()

    def get_metric(self, reset=False):
        """
        返回准确率、召回率、F值评价指标
        """
        # print('TP',self._true_positives,'TN',self._true_negatives,'FP',self._false_positives,'FN',self._false_negatives)

        precision = float(self._true_positives) / float(self._true_positives + self._false_positives + 1e-13)
        recall = float(self._true_positives) / float(self._true_positives + self._false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        if reset:
            self.reset()
        return {"precision":precision, "recall": recall, "f1_measure":f1_measure}

    def reset(self):
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0


class NerF1Measure(Metric):
    def __init__(self, label_vocab):
        self.golden_num = 0.0
        self.predict_num = 0.0
        self.right_num = 0.0
        self.label_vocab = label_vocab

    def reset(self):
        """
        重置内部状态
        """
        self.golden_num = 0.0
        self.predict_num = 0.0
        self.right_num = 0.0

    def get_metric(self, reset=False):
        """
        返回metric的指标
        """
        if self.predict_num == 0.0:
            precision = -1
        else:
            precision = (self.right_num+0.0)/self.predict_num

        if self.golden_num == 0.0:
            recall = -1
        else:
            recall = (self.right_num+0.0)/self.golden_num

        if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
            f_measure = -1
        else:
            f_measure = 2*precision*recall/(precision+recall)

        if reset:
            self.reset()

        return {"precision":precision, "recall": recall, "f1_measure":f_measure}

    def update(self, gold_matrix, pred_matrix):
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        self.golden_num += len(gold_matrix)
        self.predict_num += len(pred_matrix)
        self.right_num += len(right_ner)

    def __call__(self,
                 predictions,
                 gold_labels,
                 mask=None):
        """
        metric的抽象类

        :params predictions 预测结果的tensor
        :params gold_labels 实际结果的tensor
        :mask   mask
        """
        batch_size = gold_labels.size(0)
        seq_len = gold_labels.size(1)
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels,
            mask)

        predictions = predictions.tolist()
        gold_labels = gold_labels.tolist()
        mask = mask.tolist()

        for idx in range(batch_size):
            pred = [self.label_vocab[predictions[idx][idy]] for idy in range(seq_len) if mask[idx][idy] != 0]
            gold = [self.label_vocab[gold_labels[idx][idy]] for idy in range(seq_len) if mask[idx][idy] != 0]


        gold_matrix = get_ner_BIO(gold)
        pred_matrix = get_ner_BIO(pred)
        self.update(gold_matrix, pred_matrix)





