cd ../../../../
CUDA_VISIBLE_DEVICES=0 python ./RED++/GPT-2/gpt2_large_more.py \
    --do_test \
    --load_path ./model/GPT-2++/gpt2_large/checkpoint/delta_vector.pth \
    --model_type "gpt2-large" \