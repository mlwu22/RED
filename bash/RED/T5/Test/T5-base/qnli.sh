cd ../../../../../
CUDA_VISIBLE_DEVICES=0 python ./RED/T5/t5_base.py \
    --dataset_name "qnli" \
    --do_test \
    --load_path ./model/T5/T5-base/checkpoint/qnli \
    --model_type "t5-base" \