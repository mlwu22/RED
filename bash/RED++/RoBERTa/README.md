# Introduction

This directory contains the scrpits for training and evaluation of RED++ on RoBERTa-base and RoBERTa-large, the results are shown as follows:


|           | Params. |      MNLI      | SST2 | MRPC | CoLA |      QNLI       |      QQP       | RTE  | STSB |      Avg.      |
| :-------: | :-----: | :------------: | :--: | :--: | :--: | :-------------: | :------------: | :--: | :--: | :------------: |
|   LoRA    |  0.3M   |      86.6      | 93.9 | 88.7 | 59.7 |      92.6       |      90.4      | 75.3 | 90.3 |      84.7      |
|  RED (B)  |  0.02M  |      83.9      | 93.9 | 89.2 | 61.0 |      90.7       |      87.2      | 78.0 | 90.4 |      84.3      |
| RED++ (B) |  0.09M  | 85.9$\uparrow$ | 93.9 | 89.2 | 61.0 | 90.7 $\uparrow$ | 89.1$\uparrow$ | 78.0 | 90.4 | 84.8$\uparrow$ |
|   LoRA    |  0.8M   |      90.2      | 96.0 | 89.8 | 65.5 |      94.7       |      90.7      | 86.3 | 91.7 |      88.1      |
|  RED(L)   |  0.05M  |      89.5      | 96.0 | 90.3 | 68.1 |      93.5       |      88.8      | 86.2 | 91.3 |      87.9      |
| RED++ (L) |  0.25M  | 90.6$\uparrow$ | 96.0 | 90.3 | 68.1 | 94.0$\uparrow$  | 90.2$\uparrow$ | 86.2 | 91.3 | 88.3$\uparrow$ |

Notation: "(B)" denotes RoBERTa base, "(L)" denotes RoBERTa large, "++"" denotes that we introduce more trainable parameters to train the model, and "$\uparrow$\" denotes an improvement in the performance of the model on these datasets (MNLI, QNLI and QQP)



# Evaluation

- Evaluate RED on RoBERTa-base 

  ```bash
  cd Test/RoBERTa-base
  bash run.sh
  ```



- Evaluation scripts (MNLI):

  ```bash
  cd ../../../../../
  CUDA_VISIBLE_DEVICES=3 python ./RED++/RoBERTa/roberta_large_more.py \
      --dataset_name "mnli" \
      --do_test \
      --load_path ./model/RoBERTa++/RoBERTa_large/checkpoint/mnli \
      --model_type "roberta_large" \
  ```

  - `--load_path` denotes the path to load  editing vectors



# Training

1. Training and evaluating on RoBERTa-base

   ```bash
   cd RoBERTa-base
   bash run.sh
   ```


- Training script of mnli.sh:

  ```bash
  seeds=(42)
  for seed in ${seeds[@]}
  do
  CUDA_VISIBLE_DEVICES=0 python ../../../../RED++/RoBERTa/roberta_base_more.py \
      --seed $seed \
      --weight_decay 0.0 \
      --dataset_name "mnli" \
      --batch_size 32 \
      --lr 0.001 \
      --do_train \
      --do_eval \
      --do_test \
      --warmup_rate 0.06 \
      --operation_key "more" \
      --model_type "roberta_base" \
      --epochs 20
  done
  ```

