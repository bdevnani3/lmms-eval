#!/bin/bash
#SBATCH --job-name=all_subclip_qwen25
#SBATCH --output=/nethome/bdevnani3/flash/lmms_eval_expts/qwen25_vl/all_subclip_qwen25_%j.out
#SBATCH --error=/nethome/bdevnani3/flash/lmms_eval_expts/qwen25_vl/all_subclip_qwen25_%j.out
#SBATCH --partition=hoffman-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node="a40:4"
#SBATCH --qos=long
#SBATCH --exclude=puma

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
cd /nethome/bdevnani3/flash/final_lmms_eval/lmms-eval/
conda activate final_lmms_eval
export HF_HOME=/nethome/bdevnani3/flash/.cache
export PYTHONIOENCODING=utf-8

export EXPT_NAME="subclip"

# TASKS=("videomme" "mlvu_dev" "mvbench" "egoschema_subset" "longvideobench_val_v" "tempcompass_caption_matching" "tempcompass_multi_choice" "tempcompass_yes_no")
TASKS=("nextqa_mc_test")

for TASK in "${TASKS[@]}"; do
    export TASK
    export OUTPUT_PATH="/nethome/bdevnani3/flash/.cache/results_final/llava_onevision_qwen2_5_ov/${TASK}/${EXPT_NAME}"

    echo "Running subclip for ${TASK} saving results at ${OUTPUT_PATH}"

    accelerate launch \
    --num_processes=4 \
    --main_process_port 1111 \
    -m lmms_eval  \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=12845056,use_flash_attention_2=True,interleave_visuals=False \
    --tasks ${TASK} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --im_resize_shape 16 \
    --max_frames_num 800 \
    --use_subclip_detection

    echo "Completed subclip for ${TASK}"
done

