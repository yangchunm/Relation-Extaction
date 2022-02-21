import torch
import torch.nn as nn
import torch.nn.functional as F
from .CNN import CNN
from .RNN import RNN
from .ResNet import ResNet
from .AggregationLayer import AggregationLayer


class BaseModule(nn.Module):
    def __init__(self, char_size, pos_size, cfg):
        super(BaseModule, self).__init__()
        self.cfg = cfg
        self.char_size = char_size
        self.pos_size = pos_size

        self.type_Net = cfg.type_Net
        self.rnn = RNN(char_size, pos_size, cfg)
        self.cnn = CNN(char_size, pos_size, cfg)
        self.resNet = ResNet(cfg)
        self.aggregaionLayer = AggregationLayer(cfg)

    def forward(self, sentences, poses1, poses2, head_ends, tail_starts, lengths):
        if self.type_Net in ["CNN", "CNN_Att", "PCNN"]:
            cnn_out = self.cnn(sentences, poses1, poses2, lengths)
            output = self.aggregaionLayer(cnn_out, head_ends, tail_starts)

        elif self.type_Net in ["ResNet", "ResNet_Att"]:
            cnn_out = self.cnn.get_cnn_features(sentences, poses1, poses2, lengths)
            res_out = self.resNet(cnn_out)
            output = self.aggregaionLayer(res_out, head_ends, tail_starts)

        elif self.type_Net in ["BiGRU", "BiLSTM", "BiGRU_Att", "BiLSTM_Att"]:
            _, rnn_out = self.rnn(sentences, poses1, poses2, lengths)
            output = self.aggregaionLayer(rnn_out, head_ends, tail_starts)

        # elif self.type_Net in ["ResNet_BiGRU", "ResNet_BiLSTM", "ResNet_BiGRU_Att", "ResNet_BiLSTM_Att"]:
        #     cnn_out = self.cnn.get_cnn_features(sentences, poses1, poses2, lengths)
        #     res_out = self.resNet(cnn_out)
        #     rnn_out = self.rnn.recurrent_layer(res_out, lengths)
        #     output = self.aggregaionLayer(rnn_out, head_ends, tail_starts)

        else:   # self.type_Net in ["ResGRU", "ResGRU_Att", "ResLSTM", "ResLSTM_Att"]:
            cnn_out = self.cnn.get_cnn_features(sentences, poses1, poses2, lengths)
            res_out = self.resNet(cnn_out)
            _, rnn_out = self.rnn(sentences, poses1, poses2, lengths)
            output = self.aggregaionLayer(torch.cat([res_out, rnn_out], dim=1), head_ends, tail_starts)

        res = F.softmax(output, 1)      # [B, N]
        return res
