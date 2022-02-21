import torch
import numpy as np
from preprocs import sort_batch_data_train

def eval(eval_dataloader, model, cfg):
    model.eval()

    count_predict = [0] * cfg.tag_size
    count_right = [0] * cfg.tag_size
    count_total = [0] * cfg.tag_size
    # val_acc = [0] * len(cfg.tag_size)
    val_precision = [0.0] * cfg.tag_size
    val_recall = [0.0] * cfg.tag_size
    val_f1 = [0.0] * cfg.tag_size
    pred_lables = []

    # 停止gradient计算，但并不会影响dropout和batchnorm层的行为。
    # 仅仅使用model.eval()已足够;with torch.zero_grad()则是更进一步加速和节省GPU算力和显存
    with torch.no_grad():
        for batch_sentence, batch_pos1, batch_pos2, batch_head_end, batch_tail_start, batch_label, batch_length in eval_dataloader:
            batch_sentence, batch_pos1, batch_pos2, batch_head_end, batch_tail_start, batch_label, batch_length, idx_unsort = sort_batch_data_train(
                batch_sentence, batch_pos1, batch_pos2, batch_head_end, batch_tail_start, batch_label, batch_length)

            if cfg.use_gpu:
                batch_sentence = batch_sentence.cuda()
                batch_pos1 = batch_pos1.cuda()
                batch_pos2 = batch_pos2.cuda()
                batch_length = batch_length.cuda()
                batch_label = batch_label.cuda()

            y = model(batch_sentence, batch_pos1, batch_pos2, batch_head_end, batch_tail_start, batch_length)
            pred_label = np.argmax(y.cpu().numpy(), axis=1)  # 输出最大元素的索引值
            pred_label = pred_label[idx_unsort]  # 恢复排序前的顺序
            batch_label = batch_label[idx_unsort].cpu().numpy()  # 恢复排序前的顺序

            # ---------------------------------formula_compute eval------------------------------------
            for l1, l2 in zip(pred_label, batch_label):
                count_predict[l1] += 1
                count_total[l2] += 1
                if l1 == l2:
                    count_right[l1] += 1
            for i in range(cfg.tag_size):
                if count_right[i] != 0:
                    if count_predict[i] != 0:
                        val_precision[i] = 100 * round(float(count_right[i]) / count_predict[i], 2)
                    if count_total != 0:
                        val_recall[i] = 100 * round(float(count_right[i]) / count_total[i], 2)
                if val_recall[i] != 0 and val_precision[i] != 0:
                    val_f1[i] = round(2 * val_precision[i] * val_recall[i] / (val_precision[i] + val_recall[i]), 2)
            return val_precision, val_recall, val_f1, pred_lables

            # ----------------------------------sklearn.metrics_eval----------------------------------------
            # macro注重大类的影响, micro注重小类的影响
            # val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(batch_label, pred_label, average='macro', warn_for=tuple())
            # return val_precision, val_recall, val_f1