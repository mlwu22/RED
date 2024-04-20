from datasets import load_dataset
import os

task_map={
    "sst2": "glue",
    "mrpc": "glue",
    "qqp":  "glue",
    "qnli": "glue",
    "mnli": "glue",
    "rte":  "glue",
    "stsb": "glue",
    "cola": "glue",
}

# GLUE for RoBERTa and T5
for k, v in task_map.items():
    dataset = load_dataset(v, k)
    dataset.save_to_disk(os.path.join("./data", "glue_roberta_base", k))

for k, v in task_map.items():
    dataset = load_dataset(v, k)
    dataset.save_to_disk(os.path.join("./data", "glue_roberta_large", k))

for k, v in task_map.items():
    dataset = load_dataset(v, k)
    dataset.save_to_disk(os.path.join("./data", "glue_t5_base", k))

# E2E_NLG for GPT2 
dataset = load_dataset("e2e_nlg")
dataset.save_to_disk(os.path.join("./data", "e2e_nlg"))


