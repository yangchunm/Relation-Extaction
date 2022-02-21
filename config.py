import torch
import torch.nn as nn

class Config(object):
    def __init__(self, type_Net, cls_type, cls2rel):
        self.cls_type = cls_type
        self.cls2rel = cls2rel
        self.type_Net = type_Net

        self.status = "predict"     # predict or train

        if self.type_Net == "PCNN":
            self.pool = "Piecewise-Max"
        else:
            self.pool = "Avg"

        if self.type_Net in ["CNN_Att", "ResNet_Att", "BiGRU_Att", "BiLSTM_Att", "ResGRU_Att"]:
            self.use_att = True
        else:
            self.use_att = False

        # self.limits = {"Disease-Position": 304,
        #                "Symptom-Position": 700,
        #                "Test-Disease": 342,
        #                "Test-Position": 800,
        #                "Test-Symptom": 150,
        #                "Treatment-Disease": 400,
        #                "Treatment-Position": 128}

        # batch_parameter
        self.epochs = 60
        self.batch_size = 64
        self.char_embedding_dim = 300
        self.pos_embedding_dim = 25
        self.embedding_dim = self.char_embedding_dim + self.pos_embedding_dim * 2

        # rnn_parameter
        self.rnn_num_layers = 3
        self.rnn_hidden_dim = 256

        # cnn_parameter
        self.cnn_in_channels = self.embedding_dim
        self.kernel_sizes = [3, 5, 7]   # [3, 5, 7]
        self.cnn_out_channels = 32
        self.cnn_out_H = len(self.kernel_sizes) * self.cnn_out_channels

        # aggregation_layer
        self.top_k = 5
        self.chunks = 5

        if self.type_Net in ["CNN", "CNN_Att", "ResNet_Att"]:
            self.encoder_out = self.cnn_out_H
        elif self.type_Net == "PCNN":
            self.encoder_out = self.cnn_out_H*3
        # elif self.type_Net in ["BiGRU", "BiLSTM", "ResNet_BiGRU", "BiGRU_Att", "BiLSTM_Att",
        # "ResNet_BiLSTM", "ResNet_BiGRU_Att", "ResNet_BiLSTM_Att"]:
        elif self.type_Net in ["BiGRU_Att", "BiLSTM_Att"]:
            self.encoder_out = self.rnn_hidden_dim
        else:   # self.type_Net in ["ResGRU", "ResLSTM", "ResGRU_Att"]
            self.encoder_out = self.cnn_out_H + self.rnn_hidden_dim

        if type_Net in ["CNN", "ResNet",
                        "BiGRU", "BiLSTM",
                        "ResNet_BiGRU", "ResNet_BiLSTM",
                        "ResGRU", "ResLSTM"]:
            self.aggregation = "Avg"
        elif type_Net == "PCNN":
            self.aggregation = "Piecewise-Max"
        else:
            self.aggregation = "Att"

        # other_paremeter
        self.dropout = 0.5
        self.learning_rate = 0.015
        self.use_pretrained = True
        self.use_gpu = True

        self.rel2idx = self.cls2rel[self.cls_type]
        self.tag_size = len(self.rel2idx)
        self.idx2rel = {self.rel2idx[rel]: rel for rel in self.rel2idx.keys()}

        # loaded_path
        self.pretrained_path = "./pretrained/char_"+str(self.char_embedding_dim)+".npy"

        # original paths
        self.original_train_path = "./data/original_data/" + self.cls_type + "_train.pkl"
        self.original_predict_path = "./data/original_data/" + self.cls_type + "_predict.pkl"
        self.original_model_path = "./model/original_model/" + self.cls_type + "_" + self.type_Net + ".pth"

        # extended paths
        self.extended_train_path = "./data/extended_data/" + self.cls_type + "_train.pkl"
        self.extended_predict_path = "./data/extended_data/" + self.cls_type + "_predict.pkl"
        self.extended_model_path = "./model/extended_model/" + self.cls_type + "_" + self.type_Net + ".pth"

        self.relations_saved_path = "./pred_relations/" + self.cls_type + "_relations.pkl"

        self.predict_results_path = "./data/predict_results/" + self.cls_type + ".pkl"
        self.labeled_relations_path = "./KG/labeled_relations/" + self.cls_type + ".pkl"
    def list_all_members(self):
            members = []
            for i in vars(self).items():
                members.append(i)
            # return members[:12]
            return members

