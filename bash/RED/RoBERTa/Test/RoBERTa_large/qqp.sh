cd ../../../../../
CUDA_VISIBLE_DEVICES=0 python ./RED/RoBERTa/roberta_large.py \
    --dataset_name "qqp" \
    --do_test \
    --load_path ./model/RoBERTa/RoBERTa_large/checkpoint/qqp \
    --model_type "roberta_large" \
