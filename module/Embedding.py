import torch
import torch.nn as nn
import numpy as np


class Embedding(nn.Module):
    def __init__(self, char_size, pos_size, cfg):
        super(Embedding, self).__init__()
        self.char_size = char_size
        self.pos_size = pos_size
        self.tag_size = cfg.tag_size

        self.char_embedding_dim = cfg.char_embedding_dim
        self.pos_embedding_dim = cfg.pos_embedding_dim
        self.use_pretrained = cfg.use_pretrained
        self.pretrained_path = cfg.pretrained_path

        # embedding
        self.char_embedds = nn.Embedding(self.char_size, self.char_embedding_dim, padding_idx=0)
        if self.use_pretrained:
            self.char_embedds.weight.data.copy_(torch.from_numpy(np.load(self.pretrained_path)))
        self.pos_embedds = nn.Embedding(self.pos_size, self.pos_embedding_dim, padding_idx=0)

    def forward(self, sentences, poses1, poses2, lengths):
        """
        return: embedds = [B, L, e_H],
                mask = [B, 1, L]
        """
        sentences = sentences[:, :torch.max(lengths)]
        mask = (sentences != 0).unsqueeze(1)
        poses1 = poses1[:, :torch.max(lengths)]
        poses2 = poses2[:, :torch.max(lengths)]
        embedds = torch.cat((self.char_embedds(sentences), self.pos_embedds(poses1), self.pos_embedds(poses2)), 2)
        return embedds, mask