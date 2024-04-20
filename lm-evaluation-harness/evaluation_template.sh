CUDA_VISIBLE_DEVICES=0 proxychains4 lm_eval --model hf-auto \
    --model_args pretrained=/data/lwh/wxh/RAHF/model/HIR/HIR \
    --tasks hellaswag \
    --num_fewshot 10 \
    --device cuda \
    --batch_size 8

CUDA_VISIBLE_DEVICES=0 proxychains4 lm_eval --model hf-auto \
    --model_args pretrained=/data/lwh/wxh/RAHF/model/HIR/HIR \
    --tasks mmlu \
    --num_fewshot 5 \
    --device cuda \
    --batch_size 8

CUDA_VISIBLE_DEVICES=0 proxychains4 lm_eval --model hf-auto \
    --model_args pretrained=/data/lwh/wxh/RAHF/model/HIR/HIR \
    --tasks ai2_arc \
    --num_fewshot 25 \
    --device cuda \
    --batch_size 8

CUDA_VISIBLE_DEVICES=0 proxychains4 lm_eval --model hf-auto \
    --model_args pretrained=/data/lwh/wxh/RAHF/model/HIR/HIR \
    --tasks truthfulqa_mc2 \
    --device cuda \
    --batch_size 8

CUDA_VISIBLE_DEVICES=0 proxychains4 lm_eval --model hf-auto \
    --model_args pretrained=/data/lwh/wxh/RAHF/model/HIR/HIR \
    --tasks winogrande \
    --num_fewshot 5 \
    --device cuda \
    --batch_size 8

CUDA_VISIBLE_DEVICES=0 proxychains4 lm_eval --model hf-auto \
    --model_args pretrained=/data/lwh/wxh/RAHF/model/HIR/HIR \
    --tasks gsm8k \
    --num_fewshot 5 \
    --device cuda \
    --batch_size 4




