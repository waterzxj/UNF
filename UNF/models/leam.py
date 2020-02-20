#coding:utf-8
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from torch.nn import CrossEntropyLoss

from models.model import Model
from modules.embedding.embedding import TokenEmbedding
from modules.module_util import mask_softmax
from modules.encoder.full_connect import FullConnectLayer


class LEAM(Model):
    def __init__(self, input_dim, vocab_size, label_nums, 
                    hidden_dim=256, ngrams=3, active=True, 
                    norm=True, coefficient=1, dropout=0.0, **kwargs):
        """
        implementation: Joint Embedding of Words and Labels for Text Classification

        params:
        """
        super(LEAM, self).__init__(input_dim, vocab_size, **kwargs)
        self.label_embedding = TokenEmbedding(input_dim, label_nums) # e * c
        self.label_nums = label_nums
        padding = int(ngrams/2)
        self.active = active
        self.norm = norm
        self.coefficient = coefficient

        assert ngrams % 2==1, "ngram should be the odd"

        if self.active:
            self.conv = torch.nn.Sequential(
                            torch.nn.Conv1d(
			        self.label_nums, self.label_nums,
			        ngrams, padding=padding),
                            torch.nn.ReLU()
                        )
        else:
            self.conv = torch.nn.Conv1d(
			    self.label_nums, self.label_nums,
			        ngrams, padding=padding)


        self.fc1 = FullConnectLayer(input_dim, hidden_dim, dropout, "relu")
        self.fc2 = nn.Linear(hidden_dim, label_nums)

    def forward(self, input, label, mask=None):
        #import pdb;pdb.set_trace()
        word_embedding = self.embedding(input) #b * s * dim
        embedding_label = self.label_embedding(label)

        label_embedding = self.label_embedding.embeddings.weight

        if mask:
            word_embedding = word_embedding * mask.unsqueeze(-1).float()

        if self.norm:
            word_embedding = normalize(word_embedding, dim=2)
            label_embedding = normalize(label_embedding, dim=1) # l * dim

        #Attention操作
        G = word_embedding @ label_embedding.transpose(0, 1) # b * s * l
        att_v = self.conv(G.transpose(1,2)) #b * l * s
        att_v = att_v.transpose(1,2) #b * s * l
        att_v = F.max_pool1d(att_v, att_v.size(2)) #b * s * 1

        att_v = mask_softmax(att_v, 1, mask) #b *s * 1
        H_enc = att_v * word_embedding #b * s * dim
        att_out = torch.sum(H_enc, 1) # b * dim

        #全连接操作
        tmp = self.fc1(att_out) #b * s * hidden_dim
        logits = self.fc2(tmp) #b * s * label_nums

        output = {}
        output["logits"] = logits

        if self.coefficient != 0:
            output["coefficient"] = self.coefficient
            tmp = self.fc1(embedding_label) # b * l * hidden_dim
            logits = self.fc2(tmp)
            reg_loss = F.cross_entropy(logits, label)
            output["regulariration_loss"] = reg_loss

        return output

    def predict(self, input, label, mask=None):
        return self.forward(input, label, mask)["logits"]
