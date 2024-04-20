import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
main_dir = "/".join(cur_path.split("/")[:-2])
sys.path.append(main_dir)

from model import ActivationModelRobertaMore
from utils import calculate_loss, compute_metrics, convert_token_ids_to_int, set_log, convert_token_ids_to_float
from data_utils import load_glue_data_final
from transformers import RobertaModel, RobertaTokenizer, Trainer, EvalPrediction, RobertaForSequenceClassification
from transformers import AdamW, get_scheduler, DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.utils.rnn as rnn
import argparse
from torch.utils.tensorboard import SummaryWriter



operation_map = {
    "ffn_select_final": ["ffn", "all", [0,1,2]],
    "ffn_attn_select_final": ["ffn_attn", "all", [0,1,2]],
    "attn_select_final": ["attn", "all", [0,1,2]],
    "ffn_select_medium": ["ffn", "all", [0,1,2,3,10,11]],
    "ffn_attn_select_medium": ["ffn_attn", "all", [0,1,2,3,10,11]],
    "attn_select_medium": ["attn", "all", [0,1,2,3,10,11]],
    "ffn_all_layer": ["ffn", "all", None],
    "ffn_attn_all_layer": ["ffn_attn", "all", None],
    "attn_all_layer": ["attn", "all", None],
    "ffn_all_layer_ln": ["ffn", "ln", None],
    "res_all_layer": ["res", "all", None],
    "res_select_medium": ["res", "all", [0,1,10,11]],
    "res_with_attn_all_layer": ["res_with_attn", "all", None],
    "more" : ["more", "all", None]
}

metric_map = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "accuracy",
    "qnli": "accuracy",
    "qqp": "accuracy",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson"
}

# hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=1e-3)
parser.add_argument("--weight_decay", default=0.0)
parser.add_argument("--dataset_name", default="cola")
parser.add_argument("--model_type", default="roberta_base")
parser.add_argument("--operation_key", default="more")
parser.add_argument("--warmup_rate", default=0.06)
parser.add_argument("--batch_size", default=32)
parser.add_argument("--seed", default=42)
parser.add_argument("--epochs", default=40)
parser.add_argument("--load_path", default=None, help="Path to load trained model")
parser.add_argument("--do_train", default=False, action='store_true')
parser.add_argument("--do_eval", default=False, action='store_true')
parser.add_argument("--do_test", default =False, action='store_true')
args = parser.parse_args()

operation_key = args.operation_key
operation = operation_map[operation_key]
op_position = operation[0]
layer_type = operation[1]
exclude_layers = operation[2]

per_device_train_batch_size = int(args.batch_size)
dataset_name = args.dataset_name
lr = float(args.lr)    

weight_decay = float(args.weight_decay)
seed = int(args.seed)
torch.manual_seed(seed)
model_type = args.model_type
warmup_rate = float(args.warmup_rate)
do_train = bool(args.do_train)
do_eval = bool(args.do_eval)
do_test = bool(args.do_test)
load_path = args.load_path

meta_dir = f"{main_dir}/Results/RED++/roberta_base_more/[Substitute_model_type]/[Substitute_meta]/[Substitute_dataset]/lr_{str(lr)}/weight_dacay_{str(weight_decay)}"
meta_dir = meta_dir.replace("[Substitute_model_type]", operation_key)
log_dir = meta_dir.replace("[Substitute_meta]", "logdir")
save_dir = meta_dir.replace("[Substitute_meta]", "save_models")
record_dir = meta_dir.replace("[Substitute_meta]", "record")
epochs = int(args.epochs)

logger = set_log(log_dir, dataset_name)
logger.info(f"Args:\n" 
            f"lr: {str(lr)},\n"
            f"weight_decay: {str(weight_decay)},\n" 
            f"warmup_rate: {str(warmup_rate)},\n" 
            f"dataset_name: {dataset_name},\n" 
            f"model_type: {model_type},\n" 
            f"bs: {per_device_train_batch_size},\n" 
            f"peft_type: RED_More,\n" 
            f"seed: {str(seed)},\n" 
            f"operation_key: {operation_key},\n" 
            f"epoch: {str(epochs)}\n"
            f"load_path: {str(load_path)}\n"
            f"do_train: {do_train}\n"
            f"do_eval: {do_eval}\n"
            f"do_test: {do_test}"
            )
record_dir = record_dir.replace("[Substitute_dataset]", dataset_name)
writer = SummaryWriter(log_dir = record_dir)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
base_model = RobertaModel.from_pretrained("roberta-base", add_pooling_layer=False)

train_dataset, eval_dataset, test_dataset, num_labels = load_glue_data_final(
    tokenizer = tokenizer, dataset_name = dataset_name, model_type = model_type, seed=seed
    )


logger.info(f'train nums: {str(len(train_dataset))}, eval nums: {str(len(eval_dataset))}, test nums: {str(len(test_dataset))}')
logger.info(f'train example: {train_dataset[0]["input_ids"]}')
activation_model = ActivationModelRobertaMore(base_model, num_labels, op_position=op_position, layer_type=layer_type, exclude_layers=exclude_layers)
logger.info(f"Current trainable parameter rate: {activation_model.print_trainable_parameters()}")

data_collator = DataCollatorWithPadding(tokenizer)
train_dataLoader = DataLoader(
                dataset = train_dataset, 
                shuffle = True,
                batch_size = per_device_train_batch_size,
                collate_fn = data_collator,
               )

eval_dataLoader = DataLoader(
                dataset = eval_dataset, 
                batch_size = per_device_train_batch_size,
                collate_fn = data_collator,
               )

test_dataLoader = DataLoader(
                dataset = test_dataset, 
                batch_size = per_device_train_batch_size,
                collate_fn = data_collator,
               )

total_train_steps = len(train_dataLoader)*epochs

optimizer = AdamW(
    params=activation_model.parameters(),
    lr = lr,
    weight_decay = weight_decay
    )

lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=total_train_steps*warmup_rate, num_training_steps=total_train_steps)

select_epoch = 0
total_step = 0
min_eval_loss = 10000
max_eval_metric = -100
activation_model.to(torch.device("cuda"))

if(do_train):
    for epoch in range(epochs):
        for step, batch in enumerate(tqdm(train_dataLoader)):
            batch = {k:v.to(torch.device("cuda")) for k,v in batch.items()}
            labels = batch["labels"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            output = activation_model(input_ids, attention_mask)
            logits = output["logits"]
            if(dataset_name=="stsb"):  
                loss = calculate_loss(logits, labels, -100, mse=True)
            else:
                loss = calculate_loss(logits, labels, -100)
            writer.add_scalar("loss", loss, total_step)
            print(f"step: {step}, train_loss: {loss}")
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_step+=1

        if(do_eval):
            activation_model.eval()
            all_labels = []
            all_predicts = []
            eval_loss = 0
            eval_step = 0
            with torch.no_grad():
                for step, batch in enumerate(tqdm(eval_dataLoader)):
                    batch = {k:v.to(torch.device("cuda")) for k,v in batch.items()}
                    labels = batch["labels"]
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    output = activation_model(input_ids, attention_mask)
                    logits = output["logits"]
                    all_labels.append(labels.to(torch.device("cpu")))
                    all_predicts.append(logits.to(torch.device("cpu")))
                    if(dataset_name=="stsb"):  
                        loss = calculate_loss(logits, labels, -100, mse=True)
                    else:
                        loss = calculate_loss(logits, labels, -100)
                    eval_loss+=loss
                    eval_step+=1

            all_l = []
            all_p = []
            for labels in all_labels:
                for label in labels:
                    all_l.append(label.unsqueeze(0))
            for predicts in all_predicts:
                for predict in predicts:
                    all_p.append(predict)
            all_labels = rnn.pad_sequence(all_l, batch_first=True, padding_value=0)
            all_predicts = rnn.pad_sequence(all_p, batch_first=True, padding_value=0)
            if(dataset_name!="stsb"):
                all_predicts = torch.argmax(all_predicts, dim=-1)
            eval_prediction = EvalPrediction(predictions=all_predicts, label_ids=all_labels)
            result = compute_metrics(eval_prediction, dataset_name=dataset_name)
            result_scalar = result[metric_map[dataset_name]]
            avg_eval_loss  = eval_loss/eval_step
            writer.add_scalar("eval_loss", avg_eval_loss, total_step)
            writer.add_scalar("eval_metric", result_scalar, total_step)
            if(result_scalar > max_eval_metric):
                max_eval_metric = result_scalar
                select_epoch = epoch+1
            logger.info(f"current_epoch: {str(epoch+1)}, eval_result: {result_scalar}, eval_loss: {avg_eval_loss}, select_epoch: {select_epoch}")
            final_save_path = save_dir.replace("[Substitute_dataset]", dataset_name)
            os.makedirs(os.path.join(final_save_path, str(epoch+1)), exist_ok=True)
            activation_model.save_model(os.path.join(final_save_path, str(epoch+1), "delta_vector.pth"))
            activation_model.train()


if(do_test):
    if(do_train):
        activation_model.load_model(os.path.join(final_save_path, str(select_epoch), "delta_vector.pth"))
    else:
        activation_model.load_model(os.path.join(load_path, "delta_vector.pth"))
    activation_model.eval()
    all_labels = []
    all_predicts = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataLoader)):
            batch = {k:v.to(torch.device("cuda")) for k,v in batch.items()}
            labels = batch["labels"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            output = activation_model(input_ids, attention_mask)
            predicts = output["logits"]
            all_labels.append(labels.to(torch.device("cpu")))
            all_predicts.append(predicts.to(torch.device("cpu")))

    all_l = []
    all_p = []
    for labels in all_labels:
        for label in labels:
            all_l.append(label.unsqueeze(0))
    for predicts in all_predicts:
        for predict in predicts:
            all_p.append(predict)

    all_labels = rnn.pad_sequence(all_l, batch_first=True, padding_value=0)
    all_predicts = rnn.pad_sequence(all_p, batch_first=True, padding_value=0)
    if(dataset_name!="stsb"):
        all_predicts = torch.argmax(all_predicts, dim=-1)
    eval_prediction = EvalPrediction(predictions=all_predicts, label_ids=all_labels)
    result = compute_metrics(eval_prediction, dataset_name=dataset_name)
    logger.info(f"final_select_epoch: {str(select_epoch)}, test result: {result}")
