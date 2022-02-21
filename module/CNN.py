import torch
import torch.nn as nn
from .Embedding import Embedding


class CNN(nn.Module):
    def __init__(self, char_size, pos_size, cfg):
        super(CNN, self).__init__()

        self.char_size = char_size
        self.pos_size = pos_size
        self.use_gpu = cfg.use_gpu    //true or false

        # embedding layer
        self.embedding = Embedding(self.char_size, self.pos_size, cfg)

        # convolution layer
        self.in_channels = cfg.cnn_in_channels
        self.kernel_sizes = cfg.kernel_sizes
        self.out_channels = cfg.cnn_out_channels
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=k,
                      padding=k//2,
                      stride=1,
                      groups=1) for k in self.kernel_sizes])
        self.activation = nn.ReLU()

    def get_cnn_features(self, sentences, poses1, poses2, lengths):
        embedds, _ = self.embedding(sentences, poses1, poses2, lengths)
        embedds = torch.transpose(embedds, 1, 2)        # [B, e_H, L]
        cnn_out = [self.activation(conv(embedds)) for conv in self.convs]
        cnn_out = torch.cat(cnn_out, dim=1)             # [B, c_H, L]
        return cnn_out

    def convolution_layer(self, x):
        """
        x:      [B, H, L]
        output: [B, c_H, L]
        """
        convs = nn.ModuleList([
            nn.Conv1d(in_channels=x.size(1),
                      out_channels=self.out_channels,
                      kernel_size=k,
                      padding=k//2,
                      stride=1,
                      groups=1) for k in self.kernel_sizes])
        bn = nn.BatchNorm1d(self.out_channels)
        activation = nn.ReLU()
        if self.use_gpu:
            convs = convs.to('cuda')
            activation = activation.to('cuda')
            bn = bn.to('cuda')
        cnn_out = torch.cat([activation(bn(conv(x))) for conv in convs], dim=1)
        return cnn_out

    def forward(self, sentences, poses1, poses2, lengths):
        """
        return: cnn_out = [B, c_H, L]
        """
        cnn_out = self.get_cnn_features(sentences, poses1, poses2, lengths)
        # 5层卷积
        # cnn_out = self.convolution_layer(cnn_out)
        # cnn_out = self.convolution_layer(cnn_out)
        # cnn_out = self.convolution_layer(cnn_out)
        # cnn_out = self.convolution_layer(cnn_out)

        return cnn_out
