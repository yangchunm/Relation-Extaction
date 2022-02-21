import time
import numpy as np
from tqdm import tqdm
from preprocs import sort_batch_data_train

def train(train_dataloader, model, criterion, optimizer, cfg):
    model.train()
    train_right = 0
    train_total = 0
    epoch_start = time.time()
    for batch_sentence, batch_pos1, batch_pos2, batch_head_end, batch_tail_start, batch_label, batch_length in train_dataloader:
        batch_sentence, batch_pos1, batch_pos2, batch_head_end, batch_tail_start, batch_label, batch_length, _ = sort_batch_data_train(
            batch_sentence, batch_pos1, batch_pos2, batch_head_end, batch_tail_start, batch_label, batch_length)

        if cfg.use_gpu:
            batch_sentence = batch_sentence.cuda()
            batch_pos1 = batch_pos1.cuda()
            batch_pos2 = batch_pos2.cuda()
            batch_length = batch_length.cuda()
            batch_label = batch_label.cuda()

        y = model(batch_sentence, batch_pos1, batch_pos2, batch_head_end, batch_tail_start, batch_length)
        # if cfg.type_Net in ["CapsNet", "BiGRU_CapsNet", "BiLSTM_CapsNet"]:
        #     loss = model.caps_loss(y, batch_label)
        # else:
        loss = criterion(y, batch_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 选择得分最高的label作为预测结果
        pred_label = np.argmax(y.cpu().data.numpy(), axis=1)
        batch_label = batch_label.cpu().numpy()
        # formula_compute eval
        for l1, l2 in zip(pred_label, batch_label):
            if l1 == l2:
                train_right += 1
            train_total += 1
    train_acc = float(train_right) / train_total * 100

    # sklearn.metrics_eval
    #  p r f1 皆为 macro，因为micro时三者相同，定义为acc
    # _, _, train_acc, _ = precision_recall_fscore_support(batch_label, pred_label, average='micro', warn_for=tuple())
    epoch_finish = time.time()
    epoch_cost = epoch_finish - epoch_start

    return epoch_cost, loss, train_acc