# encoding:utf-8

# @Author: Rilzob
# @Time: 2018/11/6 下午7:33

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable


# encode English: matrix
# decode Chinese: matrix
class EncoderDecoderModel(nn.Module):
    def __init__(self, args):
        super(EncoderDecoderModel, self).__init__()

        self.embed_en = nn.Embedding(args.en_total_words, args.embedding_size)
        self.embed_cn = nn.Embedding(args.cn_total_words, args.embedding_size)

        self.encoder = nn.LSTM(args.embedding_size, args.hidden_size, batch_first=True)
        self.decoder = nn.LSTM(args.embedding_size, args.hidden_size, batch_first=True)

        self.linear = nn.Linear(args.hidden_size, args.cn_total_words)

        self.embed_en.weight.data.uniform_(-0.1, 0.1)
        self.embed_cn.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))

    def forward(self, x, x_mask, y, hidden):
        # x: B * T tensor!
        # x_mask: B * T tensor!
        # y: B * J tensor

        # encoder
        x_embedded = self.embed_cn(x)
        _, (h, c) = self.encoder(x_embedded, hidden)
        # decoder
        y_embedded = self.embed_cn(y)
        hiddens, (h, c) = self.decoder(y_embedded, hx=(h, c))

        # hiddens: B * J * hidden_size vector
        decoded = self.linear(hiddens.view(hiddens.size(0)*hiddens.size(1), hiddens.size(2)))  # score of each word
        decoded = F.log_softmax(decoded)  # B * J * cn_total_words
        return decoded.view(hiddens.size(0), hiddens.size(1), decoded.size(1)), hiddens
        # B * J * cn_total_words
