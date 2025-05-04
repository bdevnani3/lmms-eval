#!/bin/bash

export EXPT_NAME="baseline"
export TASK="videomme"

# Get the directory where this script is located, then go one level up
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
SOURCE_FILE="$PARENT_DIR/source.sh"

# Check if source.sh exists in the parent directory
if [ -f "$SOURCE_FILE" ]; then
    echo "Found source.sh at: $SOURCE_FILE"
    source "$SOURCE_FILE"
else
    echo "Error: source.sh not found at: $SOURCE_FILE"
    exit 1
fi

echo "Running baseline for ${TASK} saving results at ${OUTPUT_PATH}"

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
    --max_frames_num 32 \
    --limit_num_examples 10



