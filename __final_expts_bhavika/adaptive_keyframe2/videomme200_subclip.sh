#!/bin/bash
#SBATCH --job-name=videomme
#SBATCH --output=/nethome/bdevnani3/flash/lmms_eval_expts/videomme/videomme_adaptive_keyframe2_200_subclip_%j.out
#SBATCH --error=/nethome/bdevnani3/flash/lmms_eval_expts/videomme/videomme_adaptive_keyframe2_200_subclip_%j.out
#SBATCH --partition=hoffman-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node="a40:4"
#SBATCH --qos=long
#SBATCH --exclude=protocol

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
cd /nethome/bdevnani3/flash/final_lmms_eval/lmms-eval/
conda activate final_lmms_eval
export HF_HOME=/nethome/bdevnani3/flash/.cache
export PYTHONIOENCODING=utf-8

export EXPT_NAME="adaptive_keyframe2_200_subclip"
export TASK="videomme"
export OUTPUT_PATH="/nethome/bdevnani3/flash/.cache/results_final/llava_onevision_qwen2_7b_ov/${TASK}/${EXPT_NAME}"

echo "Running adaptive keyframe for ${TASK} saving results at ${OUTPUT_PATH}"

accelerate launch \
    --num_processes=4 \
    --main_process_port 1111 \
    -m lmms_eval  \
    --model llava_onevision \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks ${TASK}  \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --im_resize_shape 16 \
    --max_frames_num 200 \
    --use_subclip_detection


