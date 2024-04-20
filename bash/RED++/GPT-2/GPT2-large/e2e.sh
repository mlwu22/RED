seeds=(42 43 44)
for seed in ${seeds[@]}
do
CUDA_VISIBLE_DEVICES=1 python ../../../../RED++/GPT-2/gpt2_large_more.py \
    --seed $seed \
    --weight_decay 0.0 \
    --lr 0.008 \
    --do_train \
    --do_eval \
    --do_test \
    --label_smooth 0.0 \
    --warmup_step 500 \
    --model_type "gpt2-large" \
    --batch_size 10 \
    --epochs 10
done