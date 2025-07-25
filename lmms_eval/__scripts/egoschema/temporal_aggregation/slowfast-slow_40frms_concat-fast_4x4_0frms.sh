#!/bin/bash
#SBATCH --job-name=egoschema_slowfast-slow_32frms_concat-fast_4x4_128frms
#SBATCH --output=/nethome/bdevnani3/flash/lmms_eval_expts/egoschema/temp_aggregation_16/slowfast-slow_32frms_concat-fast_4x4_128frms_%j.out
#SBATCH --error=/nethome/bdevnani3/flash/lmms_eval_expts/egoschema/temp_aggregation_16/slowfast-slow_32frms_concat-fast_4x4_128frms_%j.out
#SBATCH --partition=hoffman-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node="a40:4"
#SBATCH --qos=long

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
cd /nethome/bdevnani3/flash/final_lmms_eval/lmms-eval/
conda activate final_lmms_eval
export HF_HOME=/nethome/bdevnani3/flash/.cache
export PYTHONIOENCODING=utf-8

accelerate launch \
    --num_processes=4 \
    --main_process_port 1111 \
    -m lmms_eval  \
    --model llava_onevision \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks egoschema_subset \
    --batch_size 1 \
    --output_path /nethome/bdevnani3/flash/.cache/results/llava_onevision/egoschema/mini_temp_aggregation/slowfast-slow_32frms_concat-fast_4x4_128frms_${SLURM_JOB_ID} \
    --log_samples \
    --temporal_aggregation "slowfast-slow_32frms_concat-fast_4x4_128frms" \
    --im_resize_shape 14 \
    --max_frames_num 128