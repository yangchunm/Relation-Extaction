import torch
from module.Embedding import Embedding
import numpy as np
from preprocs import sort_batch_data_pred


def pred(pred_dataloader, model, cfg):
    model.eval()

    pred_lables = []
    probabilities = []
    with torch.no_grad():
        for batch_sentence, batch_pos1, batch_pos2, batch_head_end, batch_tail_start, batch_length in pred_dataloader:
            batch_sentence, batch_pos1, batch_pos2, batch_head_end, batch_tail_start, batch_length, idx_unsort = sort_batch_data_pred(
                batch_sentence, batch_pos1, batch_pos2, batch_head_end, batch_tail_start, batch_length)

            if cfg.use_gpu:
                batch_sentence = batch_sentence.cuda()
                batch_pos1 = batch_pos1.cuda()
                batch_pos2 = batch_pos2.cuda()
                batch_length = batch_length.cuda()

            y = model(batch_sentence, batch_pos1, batch_pos2, batch_head_end, batch_tail_start, batch_length)
            probability = np.max(y.cpu().numpy(), axis=1)
            pred_label = np.argmax(y.cpu().numpy(), axis=1)
            probability = probability[idx_unsort]   # 恢复排序前的顺序
            pred_label = pred_label[idx_unsort]     # 恢复排序前的顺序
            probabilities.extend(probability)
            pred_lables.extend(pred_label)

    return probabilities, pred_lables
