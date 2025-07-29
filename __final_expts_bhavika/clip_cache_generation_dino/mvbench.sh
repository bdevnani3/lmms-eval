#!/bin/bash
#SBATCH --job-name=mvbench
#SBATCH --output=/nethome/bdevnani3/flash/lmms_eval_expts/mvbench/mvbench_clip_cache_%j.out
#SBATCH --error=/nethome/bdevnani3/flash/lmms_eval_expts/mvbench/mvbench_clip_cache_%j.out
#SBATCH --partition=hoffman-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos=long
#SBATCH --exclude=protocol

export PYTHONUNBUFFERED=TRUE

source ~/.bashrc
cd /nethome/bdevnani3/flash/final_lmms_eval/lmms-eval/
conda activate final_lmms_eval
export HF_HOME=/nethome/bdevnani3/flash/.cache
export PYTHONIOENCODING=utf-8

export EXPT_NAME="clip_cache_generation"
# need to install flash-attn on the nvidia gpu
pip install flash-attn --no-build-isolation --verbose

# Array of all MVBench tasks
declare -a TASKS=(
    "mvbench_action_sequence"
    "mvbench_action_antonym"
    "mvbench_action_count"
    "mvbench_action_localization"
    "mvbench_action_prediction"
    "mvbench_character_order"
    "mvbench_counterfactual_inference"
    "mvbench_egocentric_navigation"
    "mvbench_episodic_reasoning"
    "mvbench_fine_grained_action"
    "mvbench_fine_grained_pose"
    "mvbench_moving_attribute"
    "mvbench_moving_count"
    "mvbench_moving_direction"
    "mvbench_object_existence"
    "mvbench_object_interaction"
    "mvbench_object_shuffle"
    "mvbench_scene_transition"
    "mvbench_state_change"
    "mvbench_unexpected_action"
)

# Loop through each task and run the accelerate launch command
for TASK in "${TASKS[@]}"
do
    export OUTPUT_PATH="/nethome/bdevnani3/flash/.cache/results_final/llava_onevision_qwen2_7b_ov/${TASK}/${EXPT_NAME}"
    export CACHE_CLIP_SIMILARITY="/nethome/bdevnani3/flash/.cache/results_final/clip_cache_similarity_dino/${TASK}"

    echo "************************************************************************"
    echo "* Generating clip cache for ${TASK}"
    echo "* Saving clip similarity at ${CACHE_CLIP_SIMILARITY}"
    echo "************************************************************************"

    # Create directories if they don't exist
    mkdir -p ${OUTPUT_PATH}
    mkdir -p ${CACHE_CLIP_SIMILARITY}

    # Use set +e to continue even if the command fails
    set +e
    accelerate launch \
        --num_processes=8 \
        --main_process_port 1111 \
        -m lmms_eval  \
        --model llava_onevision \
        --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
        --tasks ${TASK} \
        --batch_size 1 \
        --output_path ${OUTPUT_PATH} \
        --log_samples \
        --im_resize_shape 16 \
        --max_frames_num 1600 \
        --cache_clip_similarity ${CACHE_CLIP_SIMILARITY}
    
    # Store the exit status
    EXIT_STATUS=$?
    
    # Reset to exit on error for other commands
    set -e
    
    if [ $EXIT_STATUS -ne 0 ]; then
        echo "WARNING: Task ${TASK} exited with status ${EXIT_STATUS}"
        echo "Continuing with next task..."
    else
        echo "Task ${TASK} completed successfully."
    fi
    
    # Add a small delay between tasks to avoid potential resource conflicts
    sleep 5
done

echo "All MVBench tasks completed!"