seeds=(42)
for seed in ${seeds[@]}
do
CUDA_VISIBLE_DEVICES=0 python ../../../../RED/RoBERTa/roberta_base.py \
    --seed $seed \
    --weight_decay 0.0 \
    --dataset_name "mnli" \
    --batch_size 32 \
    --lr 0.001 \
    --do_train \
    --do_eval \
    --do_test \
    --warmup_rate 0.06 \
    --operation_key "ffn_all_layer" \
    --model_type "roberta_base" \
    --epochs 20
done