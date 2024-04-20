CUDA_VISIBLE_DEVICES=0 python ../../../RED/Llama/generate_alpaca_eval.py   \
    --model_path "meta-llama/Llama-2-7b-hf"  \
    --save_path "../../../Results/RED/Llama/geneartion" \
    --peft "RED" \
    --peft_path "../../../model/Llama/delta_vector.pth" \
    --is_train_return False \
    --no_repeat_ngram_size 5 \
    --repetition_penalty 1.1

