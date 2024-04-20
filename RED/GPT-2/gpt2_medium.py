import sys
import os

# Append project path
cur_path = os.path.dirname(os.path.abspath(__file__))
main_dir = "/".join(cur_path.split("/")[:-2])
sys.path.append(main_dir)

from data_utils import load_e2e_data
from transformers import GPT2Tokenizer, DataCollatorWithPadding, GPT2LMHeadModel, T5Tokenizer, DataCollatorForLanguageModeling
from transformers import AdamW, get_scheduler
from torch.utils.data import DataLoader
from model import ActivationModelGPT2
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch
from tqdm import tqdm
from utils import calculate_loss
from utils import set_log
import os
import torch.nn.utils.rnn as rnn
from transformers import GenerationConfig
from peft import PeftModel

generation_config = GenerationConfig(
    num_beams=10,
    do_sample=False,
    no_repeat_ngram_size=4,
    length_penalty=0.9
)

# hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=0.0)
parser.add_argument("--weight_decay", default=0.0)
parser.add_argument("--model_type", default="gpt2-medium")
parser.add_argument("--warmup_step", default=500)
parser.add_argument("--seed", default=42)
parser.add_argument("--batch_size", default=10)
parser.add_argument("--label_smooth", default=0.0)
parser.add_argument("--epochs", default=5)
parser.add_argument("--load_path", default=None, help="Path to load trained model")
parser.add_argument("--do_train", default=False, action='store_true')
parser.add_argument("--do_eval", default=False, action='store_true')
parser.add_argument("--do_test", default =False, action='store_true')
args = parser.parse_args()

per_device_train_batch_size = int(args.batch_size)
epochs = int(args.epochs)
print(args.lr)
print(args.weight_decay)
lr = float(args.lr)
weight_decay = float(args.weight_decay)
dataset_name="e2e"
model_type = args.model_type
seed = int(args.seed)
torch.manual_seed(seed)
label_smooth = float(args.label_smooth)
warmup_step = int(args.warmup_step)
do_train = bool(args.do_train)
do_eval = bool(args.do_eval)
do_test = bool(args.do_test)
load_path = args.load_path

meta_dir = f"{main_dir}/Results/RED/gpt2-medium/[Substitute_meta]/[Substitute_dataset]/lr_{str(lr)}/weight_dacay_{str(weight_decay)}/seed_{str(seed)}/label_smooth_{str(label_smooth)}"
log_dir = meta_dir.replace("[Substitute_meta]", "logdir")
save_dir = meta_dir.replace("[Substitute_meta]", "save_models")
record_dir = meta_dir.replace("[Substitute_meta]", "record")
generation_dir = meta_dir.replace("[Substitute_meta]", "generation")

logger = set_log(log_dir, dataset_name)
logger.info(f"Args:\n" 
            f"lr: {str(lr)},\n"
            f"weight_decay: {str(weight_decay)},\n" 
            f"warmup_step: {str(warmup_step)},\n" 
            f"dataset_name: {dataset_name},\n" 
            f"model_type: {model_type},\n" 
            f"label_smooth: {label_smooth},\n"
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

base_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token_id = 0

train_dataset, eval_dataset, test_dataset = load_e2e_data(tokenizer = tokenizer)
logger.info(f'train nums: {str(len(train_dataset))}, eval nums: {str(len(eval_dataset))}, test nums: {str(len(test_dataset))}')
logger.info(f'train example: {train_dataset[0]["input_ids"]}')
data_collator = DataCollatorWithPadding(tokenizer)

train_dataLoader = DataLoader(
                dataset = train_dataset, 
                batch_size = per_device_train_batch_size,
                shuffle=True,
                collate_fn = data_collator
               )

eval_dataLoader = DataLoader(
                dataset = eval_dataset, 
                batch_size = per_device_train_batch_size,
                collate_fn = data_collator,
               )

test_dataLoader = DataLoader(
                dataset = test_dataset, 
                batch_size = 1,
                collate_fn = data_collator,
               )

base_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
activation_model = ActivationModelGPT2(base_model)
logger.info(f"Current trainable parameter rate: {activation_model.print_trainable_parameters()}")
total_train_steps = len(train_dataLoader)*epochs

optimizer = AdamW(
    params=activation_model.parameters(),
    lr = lr,
    weight_decay = weight_decay
    )

lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=total_train_steps)

select_step = 0
total_step = 0
min_eval_loss = 10000
activation_model.to(torch.device("cuda"))

if(do_train):
    for epoch in range(epochs):
        logger.info(f"current_epoch: {str(epoch+1)}")
        for step, batch in enumerate(tqdm(train_dataLoader)):
            batch = {k:v.to(torch.device("cuda")) for k,v in batch.items()}
            labels = batch["labels"]
            input_ids = batch["input_ids"]
            output = activation_model(input_ids, labels=labels)
            logits = output["logits"]
            loss = calculate_loss(logits, labels, tokenizer.pad_token_id, shift=True, label_smoothing=0.0)
            writer.add_scalar("loss", loss, total_step)
            # loss = output.loss
            print(f"step: {step}, train_loss: {loss}")
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_step+=1

            if(do_eval):
                if(total_step%500==0):
                    eval_loss = 0
                    eval_step = 0
                    activation_model.eval()
                    with torch.no_grad():
                        for step, batch in enumerate(tqdm(eval_dataLoader)):
                            batch = {k:v.to(torch.device("cuda")) for k,v in batch.items()}
                            labels = batch["labels"]
                            input_ids = batch["input_ids"]
                            output = activation_model(input_ids)
                            logits = output["logits"]
                            loss = calculate_loss(logits, labels, tokenizer.pad_token_id, shift=True, label_smoothing=0.0)
                            eval_loss+=loss
                            eval_step+=1
                    avg_eval_loss  = eval_loss/eval_step
                    if(avg_eval_loss < min_eval_loss):
                        min_eval_loss = avg_eval_loss
                        select_step = total_step
                    logger.info(f"current_step: {str(total_step)}, eval_loss: {avg_eval_loss}, select_step: {select_step}")
                    final_save_path = save_dir.replace("[Substitute_dataset]", dataset_name)
                    os.makedirs(os.path.join(final_save_path, str(total_step)), exist_ok=True)
                    activation_model.save_model(os.path.join(final_save_path, str(total_step), "delta_vector.pth"))
                    activation_model.train()

if(do_test):
    logger.info(f"\nDecoding begins\n")
    if(do_train):
        activation_model.load_model(os.path.join(final_save_path, str(select_step), "delta_vector.pth"))
    else:
        activation_model.load_model(load_path)
    activation_model.to(torch.device("cuda"))
    count = 0
    activation_model.eval()
    all_labels_dict = {}
    all_predict_dict = {}
    output_list = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataLoader)):
            batch = {k:v.to(torch.device("cuda")) for k,v in batch.items()}
            labels = batch["labels"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            input_text = tokenizer.batch_decode(input_ids.to(torch.device("cpu")), skip_special_tokens=True)[0]
            if(input_text not in all_labels_dict.keys()):
                all_labels_dict[input_text] = []
                all_predict_dict[input_text] = []

            all_labels_dict[input_text].append(tokenizer.batch_decode(labels.to(torch.device("cpu")), skip_special_tokens=True)[0])
            
            original_length = input_ids.shape[-1]
            predicts = activation_model.generate(input_ids=input_ids, generation_config = generation_config, max_length=original_length+256)
            prediction = tokenizer.batch_decode(predicts.to(torch.device("cpu")))[0]
            all_predict_dict[input_text] = prediction.split('<|endoftext|>')[1].split('\n\n')[0].strip()
            
         
    logger.info(f"\nDecoding Finish\n")       
    os.makedirs(generation_dir, exist_ok=True)
    with open(f"{generation_dir}/label.txt", "w") as file:
        for _ , refs in all_labels_dict.items():
            for ref in refs:
                file.write(ref + '\n')
            file.write('\n')    

    with open(f"{generation_dir}/pred.txt", "w") as file:
        for _ , pred in all_predict_dict.items():
            file.write(pred + '\n')

    logger.info(f"final_select_step: {str(select_step)}")
