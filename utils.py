import torch.nn as nn
import numpy as np
from transformers import EvalPrediction, AutoTokenizer
from data_utils import glue_data_num_labels_map
import evaluate
from evaluate import load
import os
import logging


cur_path = os.path.dirname(os.path.abspath(__file__))
print(cur_path)


def calculate_loss(logits, labels, ignore_index, mse=False, shift=False, label_smoothing=0.0):
    if(mse):
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())
    else:
        if(shift):
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
        loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
    return loss

def calculate_loss_gpt(logits, labels, ignore_index, batch, length, lm_mask, label_smoothing=0.0):
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)).view(batch, length)
    loss = loss * lm_mask 
    loss = loss.sum() / (lm_mask.sum() + 0.0001)
    loss = loss.mean()
    return loss

def compute_metrics(eval_pred: EvalPrediction, dataset_name):
    main_dir = os.path.dirname(os.path.abspath(__file__))
    metric_func = evaluate.load(path=os.path.join(main_dir, "evaluate/metrics/glue"), config_name=dataset_name)
    result = metric_func.compute(predictions=eval_pred.predictions, references=eval_pred.label_ids)
    if len(result.keys()) > 1:
        result["averaged_scores"] = np.mean(list(result.values())).item()
    return result


def convert_token_ids_to_int(eval_pred: EvalPrediction, tokenizer: AutoTokenizer):
    pred_str = tokenizer.batch_decode(eval_pred.predictions, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(eval_pred.label_ids, skip_special_tokens=True)
    pred_strs = lmap(str.strip, pred_str)
    label_strs = lmap(str.strip, label_str)
    eval_pred.predictions = np.array([convert_string_to_int(pred_str) for pred_str in pred_strs])
    eval_pred.label_ids = np.array([convert_string_to_int(label_str) for label_str in label_strs])
    return eval_pred

def convert_token_ids_to_float(eval_pred: EvalPrediction, tokenizer: AutoTokenizer):
    pred_str = tokenizer.batch_decode(eval_pred.predictions, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(eval_pred.label_ids, skip_special_tokens=True)
    pred_strs = lmap(str.strip, pred_str)
    label_strs = lmap(str.strip, label_str)
    eval_pred.predictions = np.array([convert_string_to_float(pred_str) for pred_str in pred_strs])
    eval_pred.label_ids = np.array([convert_string_to_float(label_str) for label_str in label_strs])
    return eval_pred

def lmap(f, x):
    return list(map(f, x))


def convert_string_to_int(string, default=0):
    try:
        res = int(string)
        if res==0 or res==1:
            return res
        else:
            return default
    except ValueError:
        return default
    
def convert_string_to_float(string, default=0.0):
    try:
        res = float(string)
        x = np.arange(0.0, 5.1, 0.2)
        x_list = x.tolist()
        x_list = [round(x,1) for x in x_list]
        if res>=0 and res<=5:
            if res in x_list:
                return res
            else:
                return round(res, 1)
        else:
            return default
    except ValueError:
        return default
    
def set_log(log_dir, dataset_name):
    log_dir = log_dir.replace("[Substitute_dataset]", dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    final_path = f"{log_dir}/log"
    logger = logging.getLogger()
    logger.setLevel('INFO')
    control = logging.StreamHandler() 
    control.setLevel('INFO')
    fhlr = logging.FileHandler(final_path)
    logger.addHandler(fhlr)
    logger.addHandler(control)
    return logger

