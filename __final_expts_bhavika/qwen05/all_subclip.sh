#!/bin/bash
#SBATCH --job-name=all_subclip_qwen05
#SBATCH --output=/nethome/bdevnani3/flash/lmms_eval_expts/qwen05/all_subclip_qwen05_%j.out
#SBATCH --error=/nethome/bdevnani3/flash/lmms_eval_expts/qwen05/all_subclip_qwen05_%j.out
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
    export OUTPUT_PATH="/nethome/bdevnani3/flash/.cache/results_final/llava_onevision_qwen2_0.5b_ov/${TASK}/${EXPT_NAME}"

    echo "Running subclip for ${TASK} saving results at ${OUTPUT_PATH}"

    accelerate launch \
    --num_processes=4 \
    --main_process_port 1111 \
    -m lmms_eval  \
    --model llava_onevision \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-0.5b-ov" \
    --tasks ${TASK} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --im_resize_shape 16 \
    --max_frames_num 800 \
    --use_subclip_detection

    echo "Completed subclip for ${TASK}"
done



