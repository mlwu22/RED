# Introduction

This directory contains the scrpits for training and evaluation of RED and RED++

- `RED` contains the scrpits for training and evaluation of vanilla representation editing method
- `RED++` contains the scrpits for training and evaluation of variant of RED, which introduces slightly more trainable parameters to adapt to datasets with larger sizes



We conducted RED++ experiments on datasets containing a large number of training examples (MNLI, QNLI, QQP, E2E_NLG) 


|           | Params. |      MNLI      | SST2 | MRPC | CoLA |      QNLI       |      QQP       | RTE  | STSB |      Avg.      |
| :-------: | :-----: | :------------: | :--: | :--: | :--: | :-------------: | :------------: | :--: | :--: | :------------: |
|   LoRA    |  0.3M   |      86.6      | 93.9 | 88.7 | 59.7 |      92.6       |      90.4      | 75.3 | 90.3 |      84.7      |
|  RED (B)  |  0.02M  |      83.9      | 93.9 | 89.2 | 61.0 |      90.7       |      87.2      | 78.0 | 90.4 |      84.3      |
| RED++ (B) |  0.09M  | 85.9&uarr; | 93.9 | 89.2 | 61.0 | 90.7 &uarr; | 89.1&uarr; | 78.0 | 90.4 | 84.8&uarr; |
|   LoRA    |  0.8M   |      90.2      | 96.0 | 89.8 | 65.5 |      94.7       |      90.7      | 86.3 | 91.7 |      88.1      |
|  RED(L)   |  0.05M  |      89.5      | 96.0 | 90.3 | 68.1 |      93.5       |      88.8      | 86.2 | 91.3 |      87.9      |
| RED++ (L) |  0.25M  | 90.6&uarr; | 96.0 | 90.3 | 68.1 | 94.0&uarr;  | 90.2&uarr; | 86.2 | 91.3 | 88.3&uarr; |

Notation: "(B)" denotes RoBERTa base, "(L)" denotes RoBERTa large, "++" denotes that we introduce more trainable parameters to train the model, and "$\uparrow$\" denotes an improvement in the performance of the model on these datasets (MNLI, QNLI, and QQP)



|          | Params. |      BLUE       |      NIST      |     METEOR      |     ROUGE_L     |     CIDEr      |
| :------: | :-----: | :-------------: | :------------: | :-------------: | :-------------: | :------------: |
| LoRA(M)  |  0.8M   |      67.43      |      8.65      |      46.01      |      69.64      |      2.42      |
|  RED(M)  |  0.05M  |      64.86      |      8.36      |      44.99      |      67.62      |      2.28      |
| RED++(M) |  0.25M  | 66.68&uarr; | 8.53&uarr; | 46.28&uarr; | 69.63&uarr; | 2.38&uarr; |
| LoRA(L)  |  1.5M   |      68.24      |      8.76      |      46.23      |      69.92      |      2.42      |
|  RED(L)  |  0.09M  |      65.77      |      8.42      |      46.12      |      69.03      |      2.36      |
| RED++(L) |  0.46M  | 68.31&uarr; | 8.78&uarr; | 46.12&uarr; | 69.80&uarr; | 2.41&uarr; |

Notation: "(M)" denotes GPT-2 medium, "(L)" denotes GPT-2 large, "++" denotes that we introduce more trainable parameters to train the model and "$\uparrow$\" denotes an improvement in the performance of the model on these metrics.

