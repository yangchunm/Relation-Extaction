import torch
import torch.nn as nn
from .Embedding import Embedding

class RNN(nn.Module):
    def __init__(self, char_size, pos_size, cfg):
        super(RNN, self).__init__()

        self.char_size = char_size
        self.pos_size = pos_size
        self.use_gpu = cfg.use_gpu
        self.type_Net = cfg.type_Net
        self.batch_size = cfg.batch_size

        # embedding layer
        self.embedding = Embedding(self.char_size, self.pos_size, cfg)

        self.embedding_dim = cfg.embedding_dim
        self.rnn_hidden_dim = cfg.rnn_hidden_dim
        self.rnn_num_layers = cfg.rnn_num_layers
        self.dropout = cfg.dropout
        # rnn layer
        if self.type_Net in ["BiGRU", "BiGRU_Att", "ResGRU", "ResGRU_Att"]:
            self.rnn = nn.LSTM(input_size=self.embedding_dim,
                                hidden_size=self.rnn_hidden_dim//2,
                                num_layers=self.rnn_num_layers,
                                bidirectional=True,
                                batch_first=True,
                                dropout=self.dropout)
        else:  # ["BiLSTM", "BiLSTM_Att", "ResLSTM", "ResLSTM_Att"]
            self.rnn = nn.GRU(input_size=self.embedding_dim,
                              hidden_size=self.rnn_hidden_dim//2,
                              num_layers=self.rnn_num_layers,
                              bidirectional=True,
                              batch_first=True,
                              dropout=self.dropout)

    def get_rnn_features(self, sentences, poses1, poses2, lengths):
        embedds, mask = self.embedding(sentences, poses1, poses2, lengths)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedds, lengths, batch_first=True)
        packed_out, _ = self.rnn(packed)
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        rnn_out = torch.transpose(rnn_out, 1, 2)
        return mask, rnn_out

    def recurrent_layer(self, x, lengths):
        """
        x:      [B, H, L]
        return: rnn_out=[B, r_H, L]
        """
        x = torch.transpose(x, 1, 2)  # [B, H, L]->[B, L, H]
        if self.type_Net in ["ResNet_BiGRU", "ResNet_BiGRU_Att"]:
            rnn = nn.GRU(input_size=x.size(2),
                         hidden_size=self.rnn_hidden_dim // 2,
                         num_layers=self.rnn_num_layers,
                         bidirectional=True,
                         batch_first=True,
                         dropout=self.dropout)
        else:   # ["ResNet_BiLSTM", "ResNet_BiLSTM_Att"]
            rnn = nn.LSTM(input_size=x.size(2),
                          hidden_size=self.rnn_hidden_dim // 2,
                          num_layers=self.rnn_num_layers,
                          bidirectional=True,
                          batch_first=True,
                          dropout=self.dropout)
        if self.use_gpu:
            rnn = rnn.to('cuda')
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        packed_out, _ = rnn(packed)
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        rnn_out = torch.transpose(rnn_out, 1, 2)
        return rnn_out

    def forward(self, sentences, poses1, poses2, lengths):
        """
        return: mask = [B, 1, L],
                rnn_out = [B, H, L],
        """
        mask, rnn_out = self.get_rnn_features(sentences, poses1, poses2, lengths)

        return mask, rnn_out
