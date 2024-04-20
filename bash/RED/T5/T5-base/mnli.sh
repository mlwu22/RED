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