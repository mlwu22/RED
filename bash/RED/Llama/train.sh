CUDA_VISIBLE_DEVICES=0 python ../../../RED/Llama/llama.py   \
    --model_path "meta-llama/Llama-2-7b-hf" \
    --data_path  "HuggingFaceH4/ultrafeedback_binarized" \
    --output_dir "../../../Results/RED/Llama/" \
    --learning_rate 2e-5