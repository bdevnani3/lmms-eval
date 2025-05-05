#!/bin/bash

export TRANSFORMERS_CACHE="/data/jjain45/bhavika"
export HF_HOME="/data/jjain45/bhavika"
export HF_DATASETS_CACHE="/data/jjain45/bhavika"

export EXPT_NAME="baseline"
export TASK="perceptiontest_val_mc"
export OUTPUT_PATH="/data/jjain45/bhavika/results/llava_onevision/${TASK}/${EXPT_NAME}"

echo "Running baseline for ${TASK} saving results at ${OUTPUT_PATH}"

accelerate launch \
    --num_processes=8 \
    --main_process_port 1111 \
    -m lmms_eval  \
    --model llava_onevision \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks ${TASK} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --im_resize_shape 16 \
    --max_frames_num 32 


