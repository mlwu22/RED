CUDA_VISIBLE_DEVICES=0 lm_eval --model RED \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=../model/Llama/checkpoint/delta_vector.pth \
    --tasks winogrande \
    --num_fewshot 5 \
    --device cuda \
    --batch_size 8

CUDA_VISIBLE_DEVICES=0 lm_eval --model RED \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=../model/Llama/checkpoint/delta_vector.pth \
    --tasks truthfulqa_mc2 \
    --device cuda \
    --batch_size 32

CUDA_VISIBLE_DEVICES=0  lm_eval --model RED \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=../model/Llama/checkpoint/delta_vector.pth \
    --tasks gsm8k \
    --num_fewshot 5 \
    --device cuda \
    --batch_size 4

CUDA_VISIBLE_DEVICES=0  lm_eval --model RED \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=../model/Llama/checkpoint/delta_vector.pth \
    --tasks ai2_arc \
    --num_fewshot 25 \
    --device cuda \
    --batch_size 4

CUDA_VISIBLE_DEVICES=0  lm_eval --model RED \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=../model/Llama/checkpoint/delta_vector.pth \
    --tasks mmlu \
    --num_fewshot 5 \
    --device cuda \
    --batch_size 2


CUDA_VISIBLE_DEVICES=0  lm_eval --model RED \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,peft=../model/Llama/checkpoint/delta_vector.pth \
    --tasks hellaswag \
    --num_fewshot 10 \
    --device cuda \
    --batch_size 4


