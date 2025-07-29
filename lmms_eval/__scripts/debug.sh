export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
cd /nethome/bdevnani3/flash/final_lmms_eval/lmms-eval/
conda activate final_lmms_eval
export HF_HOME=/nethome/bdevnani3/flash/.cache
export PYTHONIOENCODING=utf-8
pip install flash-attn --no-build-isolation

accelerate launch \
    --num_processes=1 \
    --main_process_port 1111 \
    -m lmms_eval  \
    --model llava_onevision \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks ${TASK} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --im_resize_shape 16 \
    --max_frames_num 400 \
    --use_subclip_detection

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
    --max_frames_num 1600 \
    --limit_num_examples 10 

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
    --max_frames_num 1600 \
    --limit_num_examples 10 \
    --cache_clip_similarity /nethome/bdevnani3/flash/lmms_eval_cache/clip_similarity/videomme

# longvideobench_test_v

accelerate launch \
    --num_processes=1 \
    --main_process_port 1111 \
    -m lmms_eval  \
    --model llava_onevision \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --batch_size 1 \
    --output_path /nethome/bdevnani3/flash/.cache/results/llava_onevision/videomme/${SLURM_JOB_ID} \
    --log_samples \
    --im_resize_shape 16 \
    --max_frames_num 32 \
    --limit_num_examples 10 \
    --tasks mmbench_en_dev