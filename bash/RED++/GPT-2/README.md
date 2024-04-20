# Introduction

This directory contains the scripts for training and evaluation of RED++ on GPT-2 medium and GPT-2 large, the results are shown as follows:

|          | Params. |      BLUE       |      NIST      |     METEOR      |     ROUGE_L     |     CIDEr      |
| :------: | :-----: | :-------------: | :------------: | :-------------: | :-------------: | :------------: |
| LoRA(M)  |  0.8M   |      67.43      |      8.65      |      46.01      |      69.64      |      2.42      |
|  RED(M)  |  0.05M  |      64.86      |      8.36      |      44.99      |      67.62      |      2.28      |
| RED++(M) |  0.25M  | 66.68&uarr; | 8.53&uarr; | 46.28&uarr; | 69.63&uarr; | 2.38&uarr; |
| LoRA(L)  |  1.5M   |      68.24      |      8.76      |      46.23      |      69.92      |      2.42      |
|  RED(L)  |  0.09M  |      65.77      |      8.42      |      46.12      |      69.03      |      2.36      |
| RED++(L) |  0.46M  | 68.31&uarr; | 8.78&uarr; | 46.12&uarr; | 69.80&uarr; | 2.41&uarr; |

Notation: "(M)" denotes GPT-2 medium, "(L)" denotes GPT-2 large, "++" denotes that we introduce more trainable parameters to train the model, and "$\uparrow$\" denotes an improvement in the performance of the model on these metrics.



# Evaluation

- Evaluate based on the GPT-2 large generation results we have provided

  ```bash
  cd Test
  bash gpt2_large_evaluation.sh
  ```



- Evaluation scripts:

  ```bash
  cd ../../../../e2e-metrics
  
  ./measure_scores.py ../model/GPT-2++/gpt2_large/generation/label.txt \
                      ../model/GPT-2++/gpt2_large/generation/pred.txt \
  ```

  - `../model/GPT-2++/gpt2_large/generation/label.txt` denotes the path of the reference and `../model/GPT-2++/gpt2_large/generation/pred.txt` denotes the path of the model outputs



- Use the model we have provided and specify the decoding strategy to generate samples and save them in the `pred.txt file`

  ```bash
  cd ../../../../
  CUDA_VISIBLE_DEVICES=0 python ./RED++/GPT-2/gpt2_large_more.py \
      --do_test \
      --load_path ./model/GPT-2++/gpt2_large/checkpoint/delta_vector.pth \
      --model_type "gpt2-large" \
  ```

  - `--load_path` denotes the path of editing vectors
  - When decoding is completed, the output result of the model will be saved in folder `RED/Results/RED++/gpt2-large-more/generation`



# Training

1. Training and decoding by executing this script:

   ```bash
   # Training on GPT2-medium
   cd GPT2-medium
   bash e2e.sh
   ```

   

- Training script:

  ```bash
  seeds=(42 43 44)
  for seed in ${seeds[@]}
  do
  CUDA_VISIBLE_DEVICES=0 python ../../../../RED++/GPT-2/gpt2_medium_more.py \
      --seed $seed \
      --weight_decay 0.0001 \
      --lr 0.006 \
      --do_train \
      --do_eval \
      --do_test \
      --label_smooth 0.0 \
      --warmup_step 500 \
      --model_type "gpt2-medium" \
      --batch_size 10 \
      --epochs 5
  done
  ```

  - When Training is completed, the checkpoint and output result of the model will be saved in folder `RED/Results/RED++/gpt2-medium-more/save_models` and `RED/Results/RED++/gpt2-medium-more/generation` respectively



2. Evaluate the result

   ```bash
   cd ../../../../e2e-metrics
   
   ./measure_scores.py ['label_path'] \
                       ['pred_path'] \
   ```

   - When the previous stage of training is completed, folder `RED/Results/RED++/gpt2-medium-more/generation`  will store `label` and `pred`. 
   - Passing the corresponding path, and the testing of the model can be completed

