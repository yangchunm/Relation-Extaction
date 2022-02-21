import torch
import torch.nn as nn
import torch.nn.functional as F


class AggregationLayer(nn.Module):
    def __init__(self, cfg):
        super(AggregationLayer, self).__init__()

        self.aggregation = cfg.aggregation

        self.top_k = cfg.top_k
        self.chunks = cfg.chunks
        if self.aggregation == "Att":
            self.Q = nn.Parameter(torch.randn(cfg.batch_size, 1, cfg.encoder_out))

        self.dropout_layer = nn.Dropout(cfg.dropout)
        self.fc_layer = nn.Linear(cfg.encoder_out, cfg.tag_size)

    def pooling(self, x, head_ends, tail_starts):
        """
        input: x = [B, H, L]
        return: pooled = [B, H]
        """
        if self.aggregation == "Avg":
            pooled = nn.AvgPool1d(kernel_size=x.size(2))(x).squeeze(2)  # [B, H]
        elif self.aggregation == "1-Max":
            pooled = nn.MaxPool1d(kernel_size=x.size(2))(x).squeeze(2)  # [B, H]　
        elif self.aggregation == "K-Max":
            index = x.topk(self.top_k, dim=2)[1].sort(dim=2)[0]  # .topk()和.sort()返回的是包含值和索引的元组
            pooled = x.gather(2, index)            # [B, H, top_k]
            pooled = pooled.view(pooled.size(0), -1)    # [B, H]
        elif self.aggregation == "Chunk-Max":
            chunk_pooled = []
            for chunk_conved in x.chunk(chunks=self.chunks, dim=2):
                chunk_pooled.append(
                    nn.MaxPool1d(kernel_size=chunk_conved.size(2))(chunk_conved))  # [B, H, 1]
            pooled = torch.cat(chunk_pooled, dim=2)     # [B, H, chunks]
            pooled = pooled.view(pooled.size(0), -1)    # [B, H*chunks]
        else:   # self.pool == "Piecewise-Max":
            batch_pooled = []
            for i in range(x.size(0)):
                sent_conved = x[i:i+1, :, :]
                sep1 = min([head_ends[i], tail_starts[i]])
                sep2 = max([head_ends[i], tail_starts[i]])
                sent_pieces = [sent_conved[:, :, :sep1+1],
                               sent_conved[:, :, sep1:sep2+1],
                               sent_conved[:, :, sep2: sent_conved.size(2)+1]]

                left_middle_right = []
                for piece_conved in sent_pieces:
                    left_middle_right.append(
                        # nn.MaxPool1d(kernel_size=piece_conved.size(2))(piece_conved))  # [1, H, 1]
                        nn.AvgPool1d(kernel_size=piece_conved.size(2))(piece_conved))
                batch_pooled.append(torch.cat(left_middle_right, dim=2))    # [1, H, 3]
            pooled = torch.cat(batch_pooled, dim=0)  # [B, H, 3]
            pooled = pooled.view(pooled.size(0), -1)  # [B, H*3]
        return pooled

    def DotAttention(self, V):
        """
        input: x = [B, H, L]
        return: att_out = [B, H]
        """
        K = torch.tanh(V)  # [B, H, L]
        V = torch.transpose(V, 1, 2)  # [B, L, H]
        att_weight = F.softmax(torch.bmm(self.Q, K), 2)  # [B, 1, H] * [B, H, L] -> [B, 1, L]
        att_out = torch.bmm(att_weight, V).squeeze(1)  # [B, 1, L] * [B, L, H] ->[B, 1, H] ->[B, H]
        return att_out

    def forward(self, x, head_ends, tail_starts):
        """
        input: x = [B, H, L]
        return: output = [B, T]
        """
        if self.aggregation == "Att":
            att_out = torch.tanh(self.DotAttention(x))
            agg_out = self.dropout_layer(att_out)
        else:
            pooled = self.pooling(x, head_ends, tail_starts)
            agg_out = self.dropout_layer(pooled)

        output = self.fc_layer(agg_out)
        return output
