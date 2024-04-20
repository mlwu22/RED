seeds=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56)
for seed in ${seeds[@]}
do
CUDA_VISIBLE_DEVICES=0 python ../../../../RED/RoBERTa/roberta_base.py \
    --seed $seed \
    --weight_decay 0.0 \
    --dataset_name "rte" \
    --batch_size 32 \
    --lr 0.008 \
    --do_train \
    --do_eval \
    --do_test \
    --warmup_rate 0.06 \
    --operation_key "ffn_all_layer" \
    --model_type "roberta_base" \
    --epochs 40
done