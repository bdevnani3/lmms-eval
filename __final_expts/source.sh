export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
# TODO: Set the path to the parent of where the final_lmms_eval directory is
export ROOT_ABSOLUTE_PATH="/nethome/bdevnani3/flash"
cd ${ROOT_ABSOLUTE_PATH}/final_lmms_eval/lmms-eval/
conda activate final_lmms_eval
export HF_HOME=${ROOT_ABSOLUTE_PATH}/.cache
export PYTHONIOENCODING=utf-8
export OUTPUT_PATH="${ROOT_ABSOLUTE_PATH}/.cache/results/llava_onevision/${TASK}/${EXPT_NAME}/${SLURM_JOB_ID}"
export CACHE_CLIP_SIMILARITY="${ROOT_ABSOLUTE_PATH}/lmms_eval_cache/clip_similarity/${TASK}"
