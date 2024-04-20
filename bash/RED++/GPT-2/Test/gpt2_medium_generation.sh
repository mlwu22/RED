cd ../../../../
CUDA_VISIBLE_DEVICES=3 python ./RED++/GPT-2/gpt2_medium_more.py \
    --do_test \
    --load_path ./model/GPT-2++/gpt2_medium/checkpoint/delta_vector.pth \
    --model_type "gpt2-medium" \