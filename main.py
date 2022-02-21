import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import gc
import json

from config import Config
from data import TrainData, PredData
from preprocs import get_pos1_pos2, data_shuffle_divided, statistics
from module.BaseModule import BaseModule
from train import train
from evaluate import eval
from predict import pred

# 每次生成相同的随机数
np.random.seed(16)

def trainer(data, char2idx, pos2idx, model, cfg):
    train_data, test_data = data_shuffle_divided(get_pos1_pos2(data, cfg.status), cfg.rel2idx.keys(), cfg.limits[cfg.cls_type])
    print(cfg.type_Net, " | ", "train_data=", len(train_data), " test_data=", len(test_data))
    train_dataset = TrainData(train_data, char2idx, pos2idx, cfg.rel2idx)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=1,
                                  drop_last=True)  #drop_last=True 删除最后一个不完整的batch
    test_dataset = TrainData(test_data, char2idx, pos2idx, cfg.rel2idx)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=cfg.batch_size,
                                 shuffle=True,
                                 num_workers=1,
                                 drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(reduction='mean')      # size_average=True对每个batch的loss取平均;False相加

    best_epoch = 0
    best_score = 0.0
    best_result = {'p': [0.0]*cfg.tag_size, 'r': [0.0]*cfg.tag_size, 'f1': [0.0]*cfg.tag_size}
    all_f1 = []
    for epoch in range(cfg.epochs):
        # print(cfg.cls_type, " | Start epoch: {:d}".format(epoch+1))
        # ---------------------------------------------train---------------------------------------------
        epoch_cost, loss, train_acc = train(train_dataloader, model, criterion, optimizer, cfg)
        # print("Train | Time={:.2f}s, avg_loss={:.2f}, train_acc={:.2f}"
        #         .format(epoch_cost, loss.cpu().item(), train_acc))

        # ---------------------------------------------test---------------------------------------------
        val_p, val_r, val_f1, _ = eval(test_dataloader, model, cfg)
        # f1曲线：不计算unknown
        all_f1.append(sum(val_f1[1:]) / len(val_f1[1:]))
        avg_f1 = sum(val_f1) / len(val_f1)
        if avg_f1 > best_score:
            best_score = avg_f1
            best_result = {"p": val_p, "r": val_r, "f1": val_f1}
            best_epoch = epoch
            torch.save(model.state_dict(), cfg.original_model_path)
        # print("Test  | average_result: p={:.2f}, r={:.2f}, f={:.2f}"
        #       .format(avg_p, avg_r, avg_f1))
        # print("Best_epoch: {:d}, best_f1: ".format(best_epoch), best_result["f1"])
    print(cfg.type_Net, " | Best_epoch: {:d}, best_f1: ".format(best_epoch), best_result["f1"])
    return best_result["f1"], all_f1


def prediction(pred_data, char2idx, pos2idx, model, cfg):
    if cfg.predict_type == "file":
        pred_dataset = PredData(get_pos1_pos2(pred_data, cfg.status), char2idx, pos2idx)
        pred_dataloader = DataLoader(pred_dataset,
                                     batch_size=cfg.batch_size,
                                     shuffle=True,
                                     num_workers=1,
                                     drop_last=True)
        model.load_state_dict(torch.load(cfg.original_model_path))
        _, pred_lables = pred(pred_dataloader, model, cfg)
        pred_relations = [cfg.idx2rel[lable] for lable in pred_lables]
        predicted_results = []
        for relation, data in zip(pred_relations, pred_data):
            if relation != "unknown":
                # predicted_results.append({'en1': data['en1']['word'], 'rel': relation, 'en2': data['en2']['word']})
                predicted_results.append((data['en1']['word'], relation,  data['en2']['word']))
        # return list(set(predicted_results))
        return predicted_results


cls2rel = {"Disease-Position": {"unknown": 0, "DAP": 1},
           "Symptom-Position": {"unknown": 0, "SAP": 1, "SNAP": 2},
           "Test-Disease": {"unknown": 0, "TeRD": 1},
           "Test-Position": {"unknown": 0, "TeAP": 1, "TeCP": 2},
           "Test-Symptom": {"unknown": 0, "TeRS": 1, "TeAS": 2},
           "Treatment-Disease": {"unknown": 0, "TrAD": 1, "TrRD": 2},
           "Treatment-Position": {"unknown": 0, "TrAP": 1}}         # "unknown"表示属于关系大类但不存在关系的（实体对+句子）
# -------------------------------------------------------------first train------------------------------------------------------------
char2idx = pickle.load(open("./data/char2idx.pkl", "rb"))
pos2idx = pickle.load(open("./data/pos2idx.pkl", "rb"))

for cls_type in cls2rel.keys():
# for cls_type in ["Treatment-Position"]:
    print("start ", cls_type, "......")
    for type_Net in ["CNN", "PCNN", "ResNet", "BiGRU", "BiLSTM", "ResGRU",
        "CNN_Att", "ResNet_Att", "BiGRU_Att", "BiLSTM_Att", "ResGRU_Att"]:
        cfg = Config(type_Net, cls_type, cls2rel)

        model = BaseModule(len(char2idx), len(pos2idx), cfg)
        if cfg.use_gpu:
            model = model.cuda()

        if cfg.status == "train":
            best_f1, all_f1 = trainer(pickle.load(open(cfg.original_train_path, "rb")), char2idx, pos2idx, model, cfg)
            while 0.0 in best_f1 or 100.0 in best_f1:
                print("有异常结果，重新训练！")
                best_f1, all_f1 = trainer(pickle.load(open(cfg.original_train_path, "rb")), char2idx, pos2idx, model, cfg)
        else:
            if cfg.predict_type == "file":
                predicted_results = prediction(pickle.load(open(cfg.original_predict_path, "rb")), char2idx, pos2idx, model, cfg)
                train_relations = pickle.load(open(cfg.original_train_path, "rb"))
                # print(type_Net, " | train=", len(train_relations), ", predict=", len(predicted_results))
                pickle.dump(predicted_results, (open(cfg.predict_results_path, "wb")))
                # 统计未标注数据集中存在的关系数量
                # relations_count = statistics(predicted_results, cls2rel[cls_type].keys())
                # print(type_Net, " | relations=", relations_count)
            else:
                pass
gc.collect()

# ------------------------------------------------------relation extraction---------------------------------------------------
# 使用最终的模型对最终的数据进行关系抽取，得到的结果用于构建知识图谱
for cls_type in cls2rel.keys():
    cfg = Config("ResGRU_Att", cls_type, cls2rel)
    predict_relations = pickle.load((open(cfg.predict_results_path, "rb")))
    train_relations = []
    for data in pickle.load(open(cfg.extended_train_path, "rb")):
        if data['rel'] != "unknown":
            train_relations.append((data['en1']['word'], data['rel'], data['en2']['word']))
    train_relations = list(set(train_relations))     # 去重
    labeled_relations = train_relations + predict_relations
    pickle.dump(labeled_relations, open(cfg.labeled_relations_path, "wb"))
    pickle.dump(labeled_relations, open("./KG/new/all_relations/" + cls_type + ".pkl", "wb"))
    # print("train=", len(train_relations), ", predict=", len(predict_relations), ", labeled=", len(labeled_relations))
