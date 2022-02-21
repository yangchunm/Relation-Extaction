import torch
from torch.utils import data


class TrainData(data.Dataset):
    def __init__(self, data, char2idx, pos2idx, rel2idx):
        self.data = data
        self.max_len = max([len(d['sent']) for d in data])
        self.char2idx = char2idx
        self.pos2idx = pos2idx
        self.rel2idx = rel2idx

    # 所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = list(self.data[index]['sent'])
        pos1 = self.data[index]['pos1']
        pos2 = self.data[index]['pos2']
        head_end = self.data[index]['head_end']
        tail_start = self.data[index]['tail_start']
        label = self.data[index]['label']
        pad_sentence, pad_pos1, pad_pos2 = self.padding(sentence, pos1, pos2)
        
        pad_sentence_idx = torch.LongTensor([self.char2idx[i] for i in pad_sentence])
        pad_pos1_idx = torch.LongTensor([self.pos2idx[i] for i in pad_pos1])
        pad_pos2_idx = torch.LongTensor([self.pos2idx[i] for i in pad_pos2])
        label_idx = self.rel2idx[label]
        length = len(sentence)

        return pad_sentence_idx, pad_pos1_idx, pad_pos2_idx, head_end, tail_start, label_idx, length
    
    def padding(self, sentence, pos1, pos2):
        pad_sentence = sentence + ['<PAD>']*(self.max_len-len(sentence))
        pad_pos1 = pos1 + [500]*(self.max_len-len(sentence))
        pad_pos2 = pos2 + [500]*(self.max_len-len(sentence))
        return pad_sentence, pad_pos1, pad_pos2


class PredData(data.Dataset):
    def __init__(self, data, char2idx, pos2idx):
        self.data = data
        self.max_len = max([len(d['sent']) for d in data])
        self.char2idx = char2idx
        self.pos2idx = pos2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = list(self.data[index]['sent'])
        pos1 = self.data[index]['pos1']
        pos2 = self.data[index]['pos2']
        head_end = self.data[index]['head_end']
        tail_start = self.data[index]['tail_start']
        pad_sentence, pad_pos1, pad_pos2 = self.padding(sentence, pos1, pos2)

        pad_sentence_idx = torch.LongTensor([self.char2idx[i] for i in pad_sentence])
        pad_pos1_idx = torch.LongTensor([self.pos2idx[i] for i in pad_pos1])
        pad_pos2_idx = torch.LongTensor([self.pos2idx[i] for i in pad_pos2])
        length = len(sentence)

        return pad_sentence_idx, pad_pos1_idx, pad_pos2_idx, head_end, tail_start, length

    def padding(self, sentence, pos1, pos2):
        pad_sentence = sentence + ['<PAD>'] * (self.max_len - len(sentence))
        pad_pos1 = pos1 + [500] * (self.max_len - len(sentence))
        pad_pos2 = pos2 + [500] * (self.max_len - len(sentence))

        return pad_sentence, pad_pos1, pad_pos2