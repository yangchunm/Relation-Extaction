import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import gc

from config import Config
from data import TrainData, PredData
from preprocs import get_pos1_pos2, data_shuffle_divided_bootstrapp, statistics
from module.BaseModule import BaseModule
from train import train
from evaluate import eval
from predict import pred


# 每次生成相同的随机数
np.random.seed(16)


def trainer(times, data, char2idx, pos2idx, model, cfg):
    train_data, test_data = data_shuffle_divided_bootstrapp(get_pos1_pos2(data, cfg.status), cfg.rel2idx.keys())
    train_dataset = TrainData(train_data, char2idx, pos2idx, cfg.rel2idx)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=1,
                                  drop_last=True)  # drop_last=True后可删除最后一个不完整的batch
    test_dataset = TrainData(test_data, char2idx, pos2idx, cfg.rel2idx)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=cfg.batch_size,
                                 shuffle=True,
                                 num_workers=1,
                                 drop_last=True)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(reduction='mean')  # size_average=True表示对每个batch的loss取平均;False表示相加

    best_epoch = 0
    best_score = 0.0
    best_result = {'p': [0.0] * cfg.tag_size, 'r': [0.0] * cfg.tag_size, 'f1': [0.0] * cfg.tag_size}
    for epoch in range(cfg.epochs):
        # print(cfg.cls_type, " | Start epoch: {:d}".format(epoch+1))
        # ---------------------------------------------train---------------------------------------------
        epoch_cost, loss, train_acc = train(train_dataloader, model, criterion, optimizer, cfg)
        # print("Train | Time={:.2f}s, avg_loss={:.2f}, train_acc={:.2f}"
        #         .format(epoch_cost, loss.cpu().item(), train_acc))

        # ---------------------------------------------test---------------------------------------------
        val_p, val_r, val_f1, _ = eval(test_dataloader, model, cfg)
        avg_p = sum(val_p) / len(val_p)
        avg_r = sum(val_r) / len(val_r)
        avg_f1 = sum(val_f1) / len(val_f1)
        if avg_f1 > best_score:
            best_score = avg_f1
            best_result = {"p": val_p, "r": val_r, "f1": val_f1}
            best_epoch = epoch
            torch.save(model.state_dict(), cfg.extended_model_path)
        # print("Test  | average_result: p={:.2f}, r={:.2f}, f={:.2f}"
        #       .format(avg_p, avg_r, avg_f1))
        # print("Best_epoch: {:d}, best_f1: ".format(best_epoch), best_result["f1"])
    print(cfg.cls_type, times, "| Best_epoch: {:d}, best_f1: ".format(best_epoch), best_result["f1"])
    return best_result["f1"]


def prediction(pred_data, new_train, char2idx, pos2idx, model, cfg):
    pred_dataset = PredData(get_pos1_pos2(pred_data, cfg.status), char2idx, pos2idx)
    pred_dataloader = DataLoader(pred_dataset,
                                 batch_size=cfg.batch_size,
                                 shuffle=True,
                                 num_workers=1,
                                 drop_last=True)

    model.load_state_dict(torch.load(cfg.extended_model_path))
    probabilities, pred_lables = pred(pred_dataloader, model, cfg)

    # 大于阈值的为可靠标签，小于阈值的为不可靠标签。将标签在pred_data中对应的索引值　分别保存为可靠集和不可靠集。
    Threshold = 0.7
    reliable_limit = 1000
    reliabled_idx = []
    unreliabled_idx = []
    stop_idx = len(pred_data)
    for i in range(len(pred_data)):
        if len(reliabled_idx) < reliable_limit:
            if probabilities[i] > Threshold:
                reliabled_idx.append(i)
            else:
                unreliabled_idx.append(i)
        else:
            stop_idx = i
            break
    # 如果还未遍历完pred_data,可靠集已经达到1000条，将剩下的pred_data纳入不可靠集
    if stop_idx != len(pred_data):
        unreliabled_idx.extend(range(stop_idx, len(pred_data)))

    # 根据索引值将可靠集与训练集合并保存为新的训练集;不可靠集保存为新的预测集
    for idx in reliabled_idx:
        line = pred_data[idx]
        line['rel'] = cfg.idx2rel[pred_lables[idx]]
        new_train.append(line)
    new_pred = [pred_data[idx] for idx in unreliabled_idx]

    pickle.dump(new_train, open(cfg.extended_train_path, "wb"))
    pickle.dump(new_pred, open(cfg.extended_predict_path, "wb"))
    return new_train, new_pred

# -------------------------------------------------------------main------------------------------------------------------------
cls2rel = {"Disease-Position": {"unknown": 0, "DAP": 1},
           "Symptom-Position": {"unknown": 0, "SAP": 1, "SNAP": 2},
           "Test-Disease": {"unknown": 0, "TeRD": 1},
           "Test-Position": {"unknown": 0, "TeAP": 1, "TeCP": 2},
           "Test-Symptom": {"unknown": 0, "TeRS": 1, "TeAS": 2},
           "Treatment-Disease": {"unknown": 0, "TrAD": 1, "TrRD": 2},
           "Treatment-Position": {"unknown": 0, "TrAP": 1}}  # "unknown"表示属于关系大类但不存在关系的（实体对+句子）

Nets = ["CNN", "PCNN", "ResCNN", "BiGRU", "ResGRU",
        "CNN_Att", "ResCNN_Att", "ResGRU_Att"]

char2idx = pickle.load(open("./data/char2idx.pkl", "rb"))
pos2idx = pickle.load(open("./data/pos2idx.pkl", "rb"))

# for cls_type in cls2rel.keys():
for cls_type in ["Test-Symptom"]:
    print("start ", cls_type, "......")
    for type_Net in ["CNN"]:
        cfg = Config(type_Net, cls_type, cls2rel)

        train_data = pickle.load(open(cfg.extended_train_path, "rb"))
        pred_data = pickle.load(open(cfg.extended_predict_path, "rb"))

        model = BaseModule(len(char2idx), len(pos2idx), cfg)
        if cfg.use_gpu:
            model = model.to('cuda')

        times = 0
        while len(pred_data) and times<=16:
            cfg.status = "predict"
            prediction(pred_data, train_data, char2idx, pos2idx, model, cfg)
            new_train, _ = prediction(pred_data, [], char2idx, pos2idx, model, cfg)
            statistics(new_train, cls2rel[cls_type].keys())
            print('finish predicting...')

            cfg.status = "train"
            best_f1 = trainer(times, train_data, char2idx, pos2idx, model, cfg)
            while 0.0 in best_f1 or 100.0 in best_f1:
                print("有异常结果，重新训练！")
                best_f1 = trainer(times, train_data, char2idx, pos2idx, model, cfg)
            times += 1

gc.collect()  # 释放内存