cd ../../../../../
CUDA_VISIBLE_DEVICES=0 python ./RED/RoBERTa/roberta_base.py \
    --dataset_name "mnli" \
    --do_test \
    --load_path ./model/RoBERTa/RoBERTa_base/checkpoint/mnli \
    --model_type "roberta_base" \
