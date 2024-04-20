seeds=(42 43 44)
for seed in ${seeds[@]}
do
CUDA_VISIBLE_DEVICES=2 python ../../../../RED/GPT-2/gpt2_medium.py \
    --seed $seed \
    --weight_decay 0.0001 \
    --lr 0.06 \
    --do_train \
    --do_eval \
    --do_test \
    --label_smooth 0.0 \
    --warmup_step 500 \
    --model_type "gpt2-medium" \
    --batch_size 10 \
    --epochs 5
done

