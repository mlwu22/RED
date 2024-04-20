import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
main_dir = "/".join(cur_path.split("/")[:-2])
sys.path.append(main_dir)

from datasets import load_dataset
from tqdm import tqdm 
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch
import json
from transformers import GenerationConfig,AutoConfig
import fire
from model import ActivationLLama

def main(
        model_path: str = "",
        save_path: str = "",
        peft:bool = False,
        start: int = -1,
        end: int = -1,
        peft_path:str="",
        is_train_return:bool = True,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.2,

):
    print(f"model_path: {model_path}",
          f"save_path: {save_path}",
          f"peft: {peft}",
          f"peft_path: {peft_path}",
          f"is_train_return: {is_train_return}",
          f"no_repeat_ngram_size: {no_repeat_ngram_size}",
          f"repetition_penalty: {repetition_penalty}")
    

    def process_alpacaeval(example):
        if is_train_return:
            prompt = "\n\nHuman: " + example['instruction'] + "\n\nAssistant: "
        else:
            prompt = "Human: " + example['instruction'] + "\n\nAssistant: "
        example["prompt"] = prompt
        return example


    if peft == "lora":
        model = AutoPeftModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map="auto")
    elif peft == "RED":
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map="auto")
        model = ActivationLLama(model)
        model.load_model(peft_path)
    elif peft == "ft":
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map="auto")


    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", torch_dtype=torch.bfloat16)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    dataset = load_dataset("tatsu-lab/alpaca_eval")["eval"]
    dataset = dataset.map(process_alpacaeval)


    generate_list = []
    
    generation_config = GenerationConfig(
            do_sample=False,
            no_repeat_ngram_size = no_repeat_ngram_size if no_repeat_ngram_size > 0 else 0,
            repetition_penalty= repetition_penalty,
            pad_token_id = 0,
            bos_token_id = 1,
            eos_token_id = 2,
        )

    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        if i < start and start != -1:
            continue
        if i > end and end != -1:
            break
        generate_dict = {}
        prompt = dataset[i]["prompt"]
        prompt_ids = tokenizer(prompt, return_tensors='pt').to(model.base_model.device).input_ids
        output_good = model.generate(input_ids = prompt_ids, max_new_tokens=768,generation_config=generation_config)

        completion_good = tokenizer.decode(output_good[0], skip_special_tokens=True)
        print(dataset[i]['prompt'])
        print("------------------------ good ----------------------------")
        print(completion_good.replace(prompt,""))



        generate_dict["input"] = prompt
        generate_dict["chosen"] = completion_good.replace(prompt,"")

        generate_list.append(
                {   
                "instruction":dataset[i]["instruction"],
                "output":completion_good.replace(prompt,"")
                }
            )


    with open(save_path ,"w") as f:
        json.dump(generate_list,f,indent=4)

if __name__ == "__main__":
    fire.Fire(main)