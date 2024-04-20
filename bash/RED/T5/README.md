# Introduction

This directory contains the scrpits for training and evaluation of RED on T5-base, the results are shown as follows:

![](C:%5CUsers%5CWu%20Muling%5CDesktop%5CRED%5Cbash%5CRED%5CT5%5CREADME.assets%5CResults-1713614120669.png)



# Evaluation

- Evaluate RED on T5-base

  ```bash
  cd Test/T5-base
  bash run.sh
  ```



- Evaluation scripts (MNLI):

  ```bash
  cd ../../../../../
  CUDA_VISIBLE_DEVICES=0 python ./RED/T5/t5_base.py \
      --dataset_name "mnli" \
      --do_test \
      --load_path ./model/T5/T5-base/checkpoint/mnli \
      --model_type "t5-base" \
  ```

  - `--load_path` denotes the path to load  editing vectors



We have provided checkpoints trained on three datasets, MNLI, QNLI, and QQP, for quick evaluation. However, the validation set of other datasets is small and is more affected by different random seeds when dividing the validation set and test set. Therefore, we recommend users to use the provided scripts for training and validation. The data split results are as follows:

![](C:%5CUsers%5CWu%20Muling%5CDesktop%5CRED%5Cbash%5CRED%5CT5%5CREADME.assets%5CSplit.png)



# Training

1. Training and evaluating on T5-base

   ```bash
   cd T5-base
   bash run.sh
   ```


- Training script of mnli.sh:

  ```bash
  seeds=(42)
  for seed in ${seeds[@]}
  do
  CUDA_VISIBLE_DEVICES=0 python ../../../../RED/T5/t5_base.py \
      --seed $seed \
      --weight_decay 0.0 \
      --dataset_name "mnli" \
      --batch_size 32 \
      --lr 0.05 \
      --do_train \
      --do_eval \
      --do_test \
      --warmup_rate 0.01 \
      --model_type "t5-base" \
      --epochs 10
  done
  ```




2. For tasks such as CoLA and RTE, some random seeds may cause training collapse (whether our RED method, other PEFT methods, or even FT, refer to the link [Issue](https://github.com/microsoft/LoRA/issues)) , as we also mentioned in the Appendix A section. Therefore, in order to achieve fair and effective comparison, we selected 9 random seeds for all baselines on both tasks (CoLA and RTE) and calculated the average of the highest 3 results.