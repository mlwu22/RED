import sys
import os

# Append project path
cur_path = os.path.dirname(os.path.abspath(__file__))
main_dir = "/".join(cur_path.split("/")[:-2])
sys.path.append(main_dir)



from model import ActivationModel
from utils import calculate_loss, compute_metrics, convert_token_ids_to_int, set_log, convert_token_ids_to_float
from data_utils import load_glue_data_t5
from transformers import T5ForConditionalGeneration, T5Tokenizer, EvalPrediction
from transformers import AdamW, get_scheduler, DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.utils.rnn as rnn
import argparse
from torch.utils.tensorboard import SummaryWriter


eval_strategy = "epoch"
eval_step=500


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
parser.add_argument("--weight_decay", default=0.0)
parser.add_argument("--dataset_name", default="cola")
parser.add_argument("--model_type", default="t5-base")
parser.add_argument("--lr", default=2e-2)
parser.add_argument("--warmup_rate", default=0.01)
parser.add_argument("--seed", default=42)
parser.add_argument("--eval_strategy", default="epoch")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--epochs", default=10)
parser.add_argument("--load_path", default=None, help="Path to load trained model")
parser.add_argument("--do_train", default=False, action='store_true')
parser.add_argument("--do_eval", default=False, action='store_true')
parser.add_argument("--do_test", default =False, action='store_true')
args = parser.parse_args()

per_device_train_batch_size = int(args.batch_size)
print(args.weight_decay)
weight_decay = float(args.weight_decay)
dataset_name=args.dataset_name
lr = float(args.lr)
eval_strategy = args.eval_strategy
model_type = args.model_type
warmup_rate = float(args.warmup_rate)
seed = int(args.seed)
torch.manual_seed(seed)
max_label_length = 5
do_train = bool(args.do_train)
do_eval = bool(args.do_eval)
do_test = bool(args.do_test)
load_path = args.load_path

meta_dir = f"{main_dir}/Results/RED/t5-base/[Substitute_meta]/[Substitute_dataset]/lr_{str(lr)}/weight_dacay_{str(weight_decay)}"
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
            f"peft_type: RED,\n" 
            f"seed: {str(seed)},\n" 
            f"epoch: {str(epochs)}\n"
            f"load_path: {str(load_path)}\n"
            f"do_train: {do_train}\n"
            f"do_eval: {do_eval}\n"
            f"do_test: {do_test}"
            )
record_dir = record_dir.replace("[Substitute_dataset]", dataset_name)
writer = SummaryWriter(log_dir = record_dir)

tokenizer = T5Tokenizer.from_pretrained("t5-base")
base_model = T5ForConditionalGeneration.from_pretrained("t5-base")

train_dataset, eval_dataset, test_dataset, num_labels = load_glue_data_t5(
    tokenizer = tokenizer, dataset_name = dataset_name, model_type = model_type, seed=seed
    )

logger.info(f'train nums: {str(len(train_dataset))}, eval nums: {str(len(eval_dataset))}, test nums: {str(len(test_dataset))}')
logger.info(f'train example: {train_dataset[0]["input_ids"]}')
activation_model = ActivationModel(base_model)
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
select_step = 0
total_step = 0
min_eval_loss = 10000
max_eval_metric = -1000
activation_model.to(torch.device("cuda"))

if(do_train):
    for epoch in range(epochs):
        # logger.info(f"current_epoch: {str(epoch+1)}")
        for step, batch in enumerate(tqdm(train_dataLoader)):
            batch = {k:v.to(torch.device("cuda")) for k,v in batch.items()}
            labels = batch["labels"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            output = activation_model(input_ids, attention_mask, labels=labels)
            logits = output.logits
            loss = calculate_loss(logits, labels, activation_model.base_model.config.pad_token_id)
            writer.add_scalar("loss", loss, total_step)
            # loss = output.loss
            print(f"step: {step}, train_loss: {loss}")
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_step+=1
        
            if(eval_strategy=="step" and total_step%eval_step==0 and do_eval):
                activation_model.eval()
                all_labels = []
                all_predicts = []
                with torch.no_grad():
                    for step, batch in enumerate(tqdm(eval_dataLoader)):
                        batch = {k:v.to(torch.device("cuda")) for k,v in batch.items()}
                        labels = batch["labels"]
                        input_ids = batch["input_ids"]
                        attention_mask = batch["attention_mask"]
                        predicts = activation_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_label_length)
                        all_labels.append(labels.to(torch.device("cpu")))
                        all_predicts.append(predicts.to(torch.device("cpu")))

                all_l = []
                all_p = []
                for labels in all_labels:
                    for label in labels:
                        all_l.append(label)
                for predicts in all_predicts:
                    for predict in predicts:
                        all_p.append(predict)

                all_labels = rnn.pad_sequence(all_l, batch_first=True, padding_value=0)
                all_predicts = rnn.pad_sequence(all_p, batch_first=True, padding_value=0)
                eval_prediction = EvalPrediction(predictions=all_predicts, label_ids=all_labels)
                if(dataset_name=="stsb"):
                    eval_prediction = convert_token_ids_to_float(eval_prediction, tokenizer)
                else:
                    eval_prediction = convert_token_ids_to_int(eval_prediction, tokenizer)
                result = compute_metrics(eval_prediction, dataset_name=dataset_name)
                result_scalar = result[metric_map[dataset_name]]
                writer.add_scalar("eval_metric", result_scalar, total_step)

                if(result_scalar > max_eval_metric):
                    max_eval_metric = result_scalar
                    select_step = total_step
                logger.info(f"current_step: {str(total_step)}, eval_result: {result_scalar}, select_step: {select_step}")
                final_save_path = save_dir.replace("[Substitute_dataset]", dataset_name)
                os.makedirs(os.path.join(final_save_path, str(total_step)), exist_ok=True)
                activation_model.save_model(os.path.join(final_save_path, str(total_step), "delta_vector.pth"))
                activation_model.train()

        if(eval_strategy=="epoch" and do_eval):
            activation_model.eval()
            all_labels = []
            all_predicts = []
            with torch.no_grad():
                for step, batch in enumerate(tqdm(eval_dataLoader)):
                    batch = {k:v.to(torch.device("cuda")) for k,v in batch.items()}
                    labels = batch["labels"]
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    predicts = activation_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_label_length)
                    all_labels.append(labels.to(torch.device("cpu")))
                    all_predicts.append(predicts.to(torch.device("cpu")))

            all_l = []
            all_p = []
            for labels in all_labels:
                for label in labels:
                    all_l.append(label)
            for predicts in all_predicts:
                for predict in predicts:
                    all_p.append(predict)

            all_labels = rnn.pad_sequence(all_l, batch_first=True, padding_value=0)
            all_predicts = rnn.pad_sequence(all_p, batch_first=True, padding_value=0)
            eval_prediction = EvalPrediction(predictions=all_predicts, label_ids=all_labels)
            if(dataset_name=="stsb"):
                eval_prediction = convert_token_ids_to_float(eval_prediction, tokenizer)
            else:
                eval_prediction = convert_token_ids_to_int(eval_prediction, tokenizer)
            result = compute_metrics(eval_prediction, dataset_name=dataset_name)
            result_scalar = result[metric_map[dataset_name]]
            writer.add_scalar("eval_metric", result_scalar, total_step)

            if(result_scalar > max_eval_metric):
                max_eval_metric = result_scalar
                select_epoch = epoch+1
            logger.info(f"current_epoch: {str(epoch+1)}, eval_result: {result_scalar}, select_epoch: {select_epoch}")
            final_save_path = save_dir.replace("[Substitute_dataset]", dataset_name)
            os.makedirs(os.path.join(final_save_path, str(epoch+1)), exist_ok=True)
            activation_model.save_model(os.path.join(final_save_path, str(epoch+1), "delta_vector.pth"))
            activation_model.train()

if(do_train):
    if(eval_strategy=="epoch"):
        activation_model.load_model(os.path.join(final_save_path, str(select_epoch), "delta_vector.pth"))
    elif(eval_strategy=="step"):
        activation_model.load_model(os.path.join(final_save_path, str(select_step), "delta_vector.pth"))
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
        predicts = activation_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_label_length)
        all_labels.append(labels.to(torch.device("cpu")))
        all_predicts.append(predicts.to(torch.device("cpu")))

all_l = []
all_p = []
for labels in all_labels:
    for label in labels:
        all_l.append(label)
for predicts in all_predicts:
    for predict in predicts:
        all_p.append(predict)

all_labels = rnn.pad_sequence(all_l, batch_first=True, padding_value=0)
all_predicts = rnn.pad_sequence(all_p, batch_first=True, padding_value=0)
eval_prediction = EvalPrediction(predictions=all_predicts, label_ids=all_labels)
if(dataset_name=="stsb"):
    eval_prediction = convert_token_ids_to_float(eval_prediction, tokenizer)
else:
    eval_prediction = convert_token_ids_to_int(eval_prediction, tokenizer)
result = compute_metrics(eval_prediction, dataset_name=dataset_name)
logger.info(f"Final_select_epoch: {str(select_epoch)}, test result: {result}")