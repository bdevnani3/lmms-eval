#!/bin/bash
#SBATCH --job-name=aks_all_64_32
#SBATCH --output=/nethome/bdevnani3/flash/lmms_eval_expts/AKS/aks_all_64_32_%j.out
#SBATCH --error=/nethome/bdevnani3/flash/lmms_eval_expts/AKS/aks_all_64_32_%j.out
#SBATCH --partition=overcap
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

export EXPT_NAME="AKS"

FRAMES=(32)
TASKS=("videomme" "mvbench" "mlvu_dev" "egoschema_subset" "longvideobench_val_v" "nextqa_mc_test")
# TASKS=("nextqa_mc_test")

for FRAME in "${FRAMES[@]}"; do
    for TASK in "${TASKS[@]}"; do
        export TASK
        export OUTPUT_PATH="/nethome/bdevnani3/flash/.cache/results_rebuttal/llava_onevision_qwen2_7b_ov/${TASK}/${EXPT_NAME}"

        echo "Running baseline for ${TASK} saving results at ${OUTPUT_PATH}"

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
            --max_frames_num 64 \
            --adaptive_sampling_method_max_frames ${FRAME} \
            --post_sampling_num_frames ${FRAME} \
            --use_aks 
            
        echo "##############################################################################" 
        echo "######################## Completed for ${TASK} with ${FRAME} frames ########################"
        echo "##############################################################################"
    done
done
