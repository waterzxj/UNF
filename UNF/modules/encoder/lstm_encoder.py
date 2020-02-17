#coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LstmEncoderLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, label_nums=None,
                batch_first=True, bidirectional=False, dropout=0.0):
        """"
        Lstm编码器的封装

        :params input_size 输入维度
        :params hidden_size hidden_state的维度
        :params num_layers lstm的层数
        :params batch_first 输入的第一个维度是否表示batch大小
        :params bidirectional 是否使用双向LSTM
        """
        super(LstmEncoderLayer, self).__init__()
        if bidirectional:
            lstm_size = hidden_size // 2
        else:
            lstm_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_size,
                    num_layers=num_layers, batch_first=batch_first, 
                    bidirectional=bidirectional)

        self.lstm_dropout = nn.Dropout(dropout)
        self.label_nums = label_nums

        if self.label_nums:
            self.hidden2tags = nn.Linear(hidden_size, label_nums)
    
    def forward(self, input, input_seq_lengths, batch_first=True, is_sort=False):
        """
        :params input 输入矩阵，按序列长度降序排列
        :params input_seq_length 输入序列的长度
        :params batch_first 输入矩阵的维度第一维是否是batch大小
        """
        if not is_sort:
            #对输入tensor按长度序排列
            word_seq_lengths, word_perm_idx = input_seq_lengths.sort(0, descending=True)
            input = input[word_perm_idx]

        packed_words = pack_padded_sequence(input, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = lstm_out.transpose(1, 0)
        if not is_sort:
            _, word_seq_recover = word_perm_idx.sort(0, descending=False)
            lstm_out = lstm_out[word_seq_recover]
            
        outputs = self.lstm_dropout(lstm_out) #batch * seq_len * (hidden_dim*directions)
        if self.label_nums:
            outputs = self.hidden2tags(outputs)

        return outputs
