import pickle
import os
import torch
from gensim.models import Word2Vec
import numpy as np


def get_char2idx_vocab():
    # char2idx
    char2idx = {}
    for root, dirs, files in os.walk("2020-5-17/classes"):
        for name in files:
            if ".pkl" in name:
                for line in pickle.load(open(root + "/" + name, "rb")):
                    for ch in list(line["sent"]):
                        if ch not in char2idx.keys():
                            char2idx[ch] = len(char2idx) + 1
    char2idx['<PAD>'] = 0
    pickle.dump(char2idx, open("data/char2idx.pkl", "wb"))


def word2vec_weight(char2idx, embedd_dim):
    vocab_size = len(char2idx)
    model = Word2Vec.load("pretrained/ccks_char_" + str(embedd_dim))
    weight = torch.randn(vocab_size, embedd_dim)
    for word in model.wv.index2word:
        try:
            idx=char2idx[word]      #将预训练的word 映射为char2idx中的idx
        except:
            # print(word)             #输出的是vocab中不存在的word
            continue
        weight[idx, :] = torch.from_numpy(model.wv[word])   #将vocab中存在预训练词向量的word变为预训练的词向量
    np.save("pretrained/char_" + str(embedd_dim) + ".npy", weight)


def get_pos2idx_vocab():
    # 限制句子的最大输入长度为500
    max_len = 500
    count = 0
    pos2idx = {}
    pos_vocab = []
    for root, dirs, files in os.walk("2020-5-17/classes"):
        for name in files:
            if ".pkl" in name:
                for d in pickle.load(open(root + "/" + name, "rb")):
                    sentence = d["sent"]
                    if len(sentence) <= max_len:
                        head_start = d["en1"]["pos"][0]
                        head_end = d["en1"]["pos"][1]
                        tail_start = d["en2"]["pos"][0]
                        tail_end = d["en2"]["pos"][1]
                        pos1 = [i - head_start for i in range(len(sentence))]
                        pos2 = [i - tail_start for i in range(len(sentence))]
                        for p in range(head_start, head_end):
                            pos1[p] = 0
                        for p in range(tail_start, tail_end):
                            pos2[p] = 0
                        pos_vocab += (pos1 + pos2)
                    else:
                        count += 1
    pos_vocab = sorted(list(set(pos_vocab)))
    for pos in pos_vocab:
        if pos not in pos2idx.keys():
            pos2idx[pos] = len(pos2idx) + 1
    pos2idx[500] = 0
    pickle.dump(pos2idx, open("data/pos2idx.pkl", "wb"))

# 删除长度超过500的病历
# def delete_sent():
#     max_len = 500
#     count = 0
#     for cls_type in ["Disease-Position", "Symptom-Position", "Test-Disease",
#                      "Test-Position", "Test-Symptom", "Treatment-Disease", "Treatment-Position"]:
#         data = pickle.load(open("./2020-5-17/pkl_file/" + cls_type + "_train.pkl", "rb"))
#         new_data = []
#         for line in data:
#             if len(line['sent']) <= max_len:
#                 new_data.append(line)
#             else:
#                 count += 1
#         print("delete sentence ", count)
#         pickle.dump(new_data, open("./data/original_data/" + cls_type + "_train.pkl", "wb"))
