#!/bin/bash
#SBATCH --job-name=videomme
#SBATCH --output=/nethome/bdevnani3/flash/lmms_eval_expts/videomme/videomme_smart_sampling_%j.out
#SBATCH --error=/nethome/bdevnani3/flash/lmms_eval_expts/videomme/videomme_smart_sampling_%j.out
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
    --num_processes=1 \
    --main_process_port 1111 \
    -m lmms_eval  \
    --model llava_onevision \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks videomme \
    --batch_size 1 \
    --output_path /nethome/bdevnani3/flash/.cache/results/llava_onevision/videomme/${SLURM_JOB_ID} \
    --log_samples \
    --im_resize_shape 16 \
    --max_frames_num 32 \
    --limit_num_examples 10



# videomme