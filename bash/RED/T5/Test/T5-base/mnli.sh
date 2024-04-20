cd ../../../../../
CUDA_VISIBLE_DEVICES=0 python ./RED/T5/t5_base.py \
    --dataset_name "mnli" \
    --do_test \
    --load_path ./model/T5/T5-base/checkpoint/mnli \
    --model_type "t5-base" \