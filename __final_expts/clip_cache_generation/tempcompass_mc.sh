#!/bin/bash

export TRANSFORMERS_CACHE="/data/jjain45/bhavika"
export HF_HOME="/data/jjain45/bhavika"
export HF_DATASETS_CACHE="/data/jjain45/bhavika"

export EXPT_NAME="clip_cache_generation"
export TASK="tempcompass_multi_choice"
export OUTPUT_PATH="/data/jjain45/bhavika/results/llava_onevision/${TASK}/${EXPT_NAME}"
export CACHE_CLIP_SIMILARITY="/data/jjain45/bhavika/results/llava_onevision/${TASK}/${EXPT_NAME}/clip_cache_similarity/"

# need to install flash-attn on the nvidia gpu
pip install flash-attn --no-build-isolation --verbose

echo "Generating clip cache for ${TASK} saving clip similarity at ${CACHE_CLIP_SIMILARITY}"

accelerate launch \
    --num_processes=1 \
    --main_process_port 1111 \
    -m lmms_eval  \
    --model llava_onevision \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks ${TASK} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --im_resize_shape 16 \
    --max_frames_num 1600 \
    --cache_clip_similarity ${CACHE_CLIP_SIMILARITY}



