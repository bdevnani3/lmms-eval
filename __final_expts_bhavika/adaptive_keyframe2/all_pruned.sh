#!/bin/bash
#SBATCH --job-name=all_pruned
#SBATCH --output=/nethome/bdevnani3/flash/lmms_eval_expts/adaptive_keyframe2/all_pruned_%j.out
#SBATCH --error=/nethome/bdevnani3/flash/lmms_eval_expts/adaptive_keyframe2/all_pruned_%j.out
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

export EXPT_NAME="adaptive_keyframe2"

FRAMES=(4 8)
TASKS=("videomme" "mlvu_dev" "mvbench" "egoschema_subset" "longvideobench_val_v" "tempcompass_caption_matching" "tempcompass_multi_choice" "tempcompass_yes_no")

for FRAME in "${FRAMES[@]}"; do
    for TASK in "${TASKS[@]}"; do
        export TASK
        export OUTPUT_PATH="/nethome/bdevnani3/flash/.cache/results_final/llava_onevision_qwen2_7b_ov/${TASK}/${EXPT_NAME}"

        echo "Running baseline for ${TASK} saving results at ${OUTPUT_PATH}"

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
            --max_frames_num 400 \
            --use_subclip_detection \
            --adaptive_sampling_method_max_frames ${FRAME} \
            --post_sampling_num_frames ${FRAME}
            
        echo "##############################################################################" 
        echo "######################## Completed for ${TASK} with ${FRAME} frames ########################"
        echo "##############################################################################"
    done
done
