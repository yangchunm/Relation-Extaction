import torch
import torch.nn as nn
from .CNN import CNN


class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()

        self.in_channels = cfg.cnn_out_H    # 32 * 3
        self.out_channels = cfg.cnn_out_channels

        self.convs1 = nn.ModuleList([
            nn.Conv1d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=k,
                      padding=k//2,
                      stride=1,
                      groups=1) for k in cfg.kernel_sizes])
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.convs2 = self.convs1 = nn.ModuleList([
            nn.Conv1d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=k,
                      padding=k//2,
                      stride=1,
                      groups=1) for k in cfg.kernel_sizes])
        self.bn2 = nn.BatchNorm1d(self.out_channels)
        self.activation = nn.ReLU()

    def residual_block(self, x):
        output = torch.cat([self.activation(self.bn1(conv(x))) for conv in self.convs1], dim=1)
        output = torch.cat([self.activation(self.bn1(conv(output))) for conv in self.convs2], dim=1)
        output = self.activation(output + x)
        return output

    def forward(self, cnn_out):
        """
        input:  cnn_out = [B, c_H, L]
        return: res_out = [B, c_H, L]
        """
        res_out = self.residual_block(cnn_out)
        res_out = self.residual_block(res_out)
        res_out = self.residual_block(res_out)
        res_out = self.residual_block(res_out)

        return res_out
