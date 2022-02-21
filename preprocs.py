import numpy as np
from random import shuffle

np.random.seed(16)


# 获取每个字符距离实体对的两个距离
def get_pos1_pos2(data, status):
    data_input = []
    for d in data:
        sentence = d["sent"]
        head_start = d["en1"]["pos"][0]
        head_end = d["en1"]["pos"][1]
        tail_start = d["en2"]["pos"][0]
        tail_end = d["en2"]["pos"][1]
        pos1 = [i-head_start for i in range(len(sentence))]
        pos2 = [i-tail_start for i in range(len(sentence))]
        for p in range(head_start, head_end):
            pos1[p] = 0
        for p in range(tail_start, tail_end):
            pos2[p] = 0
        if status == "train":
            label = d["rel"]
            data_input.append({"sent": sentence, "pos1": pos1, "pos2": pos2,
                               "head_end": head_end, "tail_start": tail_start,
                               "label": label})
        else:   # status == "predict":
            data_input.append({"sent": sentence, "pos1": pos1, "pos2": pos2,
                               "head_end": head_end, "tail_start": tail_start})
    return data_input


# 数据集划分
def data_shuffle_divided(data, relations, count_max_limit):
    train_data = []
    test_data = []
    lable_data_dict = {r: [] for r in relations}
    for d in data:
        lable_data_dict[d["label"]].append(d)
    for r in relations:
        tmp = lable_data_dict[r]
        if len(tmp) > count_max_limit:
            tmp = tmp[:count_max_limit]
        train_data.extend(tmp[:int(len(tmp)*0.7)])
        test_data.extend(tmp[int(len(tmp)*0.7):])
    shuffle(train_data)
    shuffle(test_data)
    return train_data, test_data

def data_shuffle_divided_bootstrapp(data, relations):
    train_data = []
    test_data = []
    lable_data_dict = {r: [] for r in relations}
    for d in data:
        lable_data_dict[d["label"]].append(d)
    count_max_limit = max([len(v) for v in lable_data_dict.values()])
    for r in relations:
        tmp = lable_data_dict[r]
        if len(tmp) > count_max_limit:
            tmp = tmp[:count_max_limit]
        train_data.extend(tmp[:int(len(tmp)*0.7)])
        test_data.extend(tmp[int(len(tmp)*0.7):])
    shuffle(train_data)
    shuffle(test_data)
    return train_data, test_data


# lstm输入为被填充的变长的序列pack_padded_sequence时，需要对句子进行排序
def sort_batch_data_train(sentences, poses1, poses2, head_ends, tail_starts, labels, lengths):
    lengths_sort, idx_sort = lengths.sort(0, descending=True)           #descending=True降序
    sentences_sort = sentences[idx_sort]
    poses1_sort = poses1[idx_sort]
    poses2_sort = poses2[idx_sort]
    head_ends_sort = head_ends[idx_sort]
    tail_starts_sort = tail_starts[idx_sort]
    labels_sort = labels[idx_sort]
    _, idx_unsort = idx_sort.sort(0, descending=False)
    return sentences_sort, poses1_sort, poses2_sort, head_ends_sort, tail_starts_sort, labels_sort, lengths_sort, idx_unsort


def sort_batch_data_pred(sentences, poses1, poses2, head_ends, tail_starts, lengths):
    lengths_sort, idx_sort = lengths.sort(0, descending=True)           #descending=True降序
    sentences_sort = sentences[idx_sort]
    poses1_sort = poses1[idx_sort]
    poses2_sort = poses2[idx_sort]
    head_ends_sort = head_ends[idx_sort]
    tail_starts_sort = tail_starts[idx_sort]
    _, idx_unsort = idx_sort.sort(0, descending=False)
    return sentences_sort, poses1_sort, poses2_sort, head_ends_sort, tail_starts_sort, lengths_sort, idx_unsort


def statistics(data, relations):
    relations_count = {r: 0 for r in relations}
    relations_count['all'] = len(data)
    for d in data:
        relations_count[d['rel']] += 1
    return relations_count
