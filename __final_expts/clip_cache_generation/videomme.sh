#!/bin/bash

export EXPT_NAME="clip_cache_generation"
export TASK="videomme"
source ../source.sh
# This is here because the flash-attn installation needs to be run on an nvidia gpu
pip install flash-attn --no-build-isolation --verbose

echo "Generating clip cache for ${TASK} saving clip similarity at ${CACHE_CLIP_SIMILARITY}"

accelerate launch \
    --num_processes=1 \
    --main_process_port 1111 \
    -m lmms_eval  \
    --model llava_onevision \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks videomme \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --im_resize_shape 16 \
    --max_frames_num 1600 \
    --cache_clip_similarity ${CACHE_CLIP_SIMILARITY}



