import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
main_dir = "/".join(cur_path.split("/")[:-2])
sys.path.append(main_dir)

import fire
from datasets import load_dataset
from transformers import  AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from transformers.trainer_callback import TrainerCallback
import os
import torch
from transformers import AutoModelForCausalLM
from model import ActivationLLama

MAX_INPUT_LENGTH = 256
# MAX_LENGTH = 768
MAX_LENGTH = 512

device_map = "auto"

def load_RED_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=True,
        use_auth_token=True,
        torch_dtype=torch.bfloat16,
    )
    model = ActivationLLama(model)
    return model

class CustomModelSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_path = os.path.join(checkpoint_path, "delta_vector.pth")

        kwargs["model"].save_model(save_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path) :
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))
        if  "model.safetensors" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "model.safetensors"))
       
def train(
        model_path: str = "",
        data_path: str = "",
        output_dir: str = "",
        learning_rate: float = 2e-5,
        num_train_epochs:  int = 3,
):
    model = load_RED_model(model_path=model_path)

    print(model)


    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "right"


    def process_ultra_preference(example):
        template = "Human: {prompt}\n\nAssistant: "
        prompt = example["prompt"]
        output = example["chosen"][1]["content"]

        example["prompt"] = template.format(prompt=prompt)
        example["prompt_length"] = len(tokenizer(example["prompt"]).input_ids)

        example["output"] = output
        
        example["text"] = example["prompt"] + example["output"] + " </s>"
        example["text_length"] = len(tokenizer(example["text"]).input_ids)
        return example

    train_data = load_dataset(data_path,"default")["train_sft"]
    train_data = train_data.map(process_ultra_preference,num_proc=8)
    train_data = train_data.filter(lambda x:x["prompt_length"] <= MAX_INPUT_LENGTH and x["text_length"] <= MAX_LENGTH)
    custom_saving_callback = CustomModelSavingCallback()

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir='logs',
        per_device_train_batch_size=2,
        gradient_accumulation_steps=64,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        num_train_epochs=num_train_epochs,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        warmup_ratio=0.1,
        report_to="none",  
        logging_strategy="steps"  
    )



    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        dataset_text_field="text",
        max_seq_length=MAX_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[custom_saving_callback],
    )

    trainer.train()

    output_dir = os.path.join(output_dir, "final_checkpoint")

    model.save_model(os.path.join(output_dir, "delta_vector.pth"))


if __name__ == "__main__":
    fire.Fire(train)