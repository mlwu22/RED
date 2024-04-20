seeds=(42 43 44 45 46 47 48 49 50)
for seed in ${seeds[@]}
do
CUDA_VISIBLE_DEVICES=0 python ../../../../RED/T5/t5_base.py \
    --seed $seed \
    --weight_decay 0.0 \
    --dataset_name "cola" \
    --batch_size 32 \
    --lr 0.04 \
    --do_train \
    --do_eval \
    --do_test \
    --warmup_rate 0.01 \
    --model_type "t5-base" \
    --epochs 60
done