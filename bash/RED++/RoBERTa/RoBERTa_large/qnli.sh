seeds=(42)
for seed in ${seeds[@]}
do
CUDA_VISIBLE_DEVICES=3 python ../../../../RED++/RoBERTa/roberta_large_more.py \
    --seed $seed \
    --weight_decay 0.0 \
    --dataset_name "qnli" \
    --batch_size 32 \
    --lr 0.001 \
    --do_train \
    --do_eval \
    --do_test \
    --warmup_rate 0.06 \
    --operation_key "more" \
    --model_type "roberta_large" \
    --epochs 10
done