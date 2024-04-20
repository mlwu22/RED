# Introduction

This is the official implementation for fine-tuning models through representation editing. We have provided trained models for quick evaluation and corresponding training code to reproduce our work.

There are several directories in this repo:

1. `bash` contains scripts both training and evaluation 
2. `model` contains checkpoints and generation results we have trained by RED for quick evaluation
3. `RED` contains the source code for training and evaluation of RED
4. `RED++` contains the source code for training and evaluation of the variant of RED
5. `lm-evaluation-harness` contains the source code for automated evaluation on Llama




# Quickstart

1. Install requirements

```bash
conda create -n RED python=3.8
conda activate RED
pip install -r requirements
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

- You also need to install Java 1.8




2. Download datasets and evaluate packages

```bash
# Download datasets
python download_dataset.py

# Download evaluate packages
git clone https://github.com/huggingface/evaluate.git
git clone https://github.com/tuetschek/e2e-metrics.git
```



3. Evaluate RED models
   Here we provide a simple script to evaluate the model we trained using RED on the E2E_NLG dataset, and you can go to the `bash` folder to learn more detailed test scripts.

```bash
cd/bash/GPT-2/Test

# Evaluate based on the GPT2 generation results we have provided
bash gpt2_medium_evaluation.sh

# Use the model we have provided and specify the decoding strategy to generate samples and save them in the pred.txt file
# And then use these generation results for evaluation
bash gpt2_medium_generation.sh
```



# Training

Here we provide a simple script for training on the RoBERTa-base model using the RED method on the STS-B dataset, and more detailed training details can be found in the `bash` folder

1. Execute scripts for training


```bash
cd bash/RED/RoBERTa/RoBERTa_base
bash stsb.sh
```



2. Training RoBERTa-base on STS-B

```bash
CUDA_VISIBLE_DEVICES=0 python ../../../../RED/RoBERTa/roberta_base.py \
    --seed $seed \
    --weight_decay 0.0 \
    --dataset_name "stsb" \
    --batch_size 32 \
    --lr 0.003 \
    --do_train \
    --do_eval \
    --do_test \
    --warmup_rate 0.06 \
    --operation_key "ffn_all_layer" \
    --model_type "roberta_base" \
    --epochs 40
```