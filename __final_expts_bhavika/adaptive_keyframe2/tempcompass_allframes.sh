#!/bin/bash
#SBATCH --job-name=tempcompass
#SBATCH --output=/nethome/bdevnani3/flash/lmms_eval_expts/tempcompass/tempcompass_adaptive_keyframe2_allframes_%j.out
#SBATCH --error=/nethome/bdevnani3/flash/lmms_eval_expts/tempcompass/tempcompass_adaptive_keyframe2_allframes_%j.out
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
export TASK="tempcompass"

# Array of MAX_FRAMES_NUM values to iterate through
FRAME_NUMS=(400 200 100)
TASKS=("tempcompass_caption_matching" "tempcompass_multi_choice" "tempcompass_yes_no")

# Iterate through each task
for TASK in "${TASKS[@]}"; do
    # Iterate through each MAX_FRAMES_NUM value
    for MAX_FRAMES_NUM in "${FRAME_NUMS[@]}"; do
        export MAX_FRAMES_NUM
        export TASK
        
        export EXPT_NAME="adaptive_keyframe2_${MAX_FRAMES_NUM}_subclip"
        export OUTPUT_PATH="/nethome/bdevnani3/flash/.cache/results_final/llava_onevision_qwen2_7b_ov/${TASK}/${EXPT_NAME}"
        
        echo "Running adaptive keyframe for ${TASK} with ${MAX_FRAMES_NUM} frames, saving results at ${OUTPUT_PATH}"
        
        accelerate launch \
            --num_processes=8 \
            --main_process_port 1111 \
            -m lmms_eval  \
            --model llava_onevision \
            --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
            --tasks ${TASK}  \
            --batch_size 1 \
            --output_path ${OUTPUT_PATH} \
            --log_samples \
            --im_resize_shape 16 \
            --max_frames_num ${MAX_FRAMES_NUM} \
            --use_subclip_detection
            
        echo "Completed run with ${MAX_FRAMES_NUM} frames for ${TASK}"
        sleep 5  # Short pause between runs
    done
done

echo "All frame number configurations completed"