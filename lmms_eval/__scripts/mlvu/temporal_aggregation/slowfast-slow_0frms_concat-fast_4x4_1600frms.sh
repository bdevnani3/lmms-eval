#!/bin/bash
#SBATCH --job-name=mlvu_slowfast-slow_0frms_concat-fast_4x4_1600frms
#SBATCH --output=/nethome/bdevnani3/flash/lmms_eval_expts/mlvu/temp_aggregation_16/slowfast-slow_0frms_concat-fast_4x4_1600frms_%j.out
#SBATCH --error=/nethome/bdevnani3/flash/lmms_eval_expts/mlvu/temp_aggregation_16/slowfast-slow_0frms_concat-fast_4x4_1600frms_%j.out
#SBATCH --partition=hoffman-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos=long

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
cd /nethome/bdevnani3/flash/final_lmms_eval/lmms-eval/
conda activate final_lmms_eval
export HF_HOME=/nethome/bdevnani3/flash/.cache
export PYTHONIOENCODING=utf-8

accelerate launch \
    --num_processes=8 \
    --main_process_port 1111 \
    -m lmms_eval  \
    --model llava_onevision \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks mlvu_dev \
    --batch_size 1 \
    --output_path /nethome/bdevnani3/flash/.cache/results/llava_onevision/mlvu/temp_aggregation_16/slowfast-slow_0frms_concat-fast_4x4_1600frms/${SLURM_JOB_ID} \
    --log_samples \
    --temporal_aggregation "slowfast-slow_0frms_concat-fast_4x4_1600frms" \
    --im_resize_shape 16 \
    --max_frames_num 1600 