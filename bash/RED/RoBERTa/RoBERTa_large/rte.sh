seeds=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56)
for seed in ${seeds[@]}
do
CUDA_VISIBLE_DEVICES=1 python ../../../../RED/RoBERTa/roberta_large.py \
    --seed $seed \
    --weight_decay 0.0001 \
    --dataset_name "rte" \
    --batch_size 32 \
    --lr 0.005 \
    --do_train \
    --do_eval \
    --do_test \
    --warmup_rate 0.01 \
    --operation_key "ffn_all_layer" \
    --model_type "roberta_large" \
    --epochs 20
done