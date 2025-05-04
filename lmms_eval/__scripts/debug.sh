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
    --tasks videomme \
    --batch_size 1 \
    --output_path /nethome/bdevnani3/flash/.cache/results/llava_onevision/videomme/${SLURM_JOB_ID} \
    --log_samples \
    --im_resize_shape 16 \
    --max_frames_num 1600 \
    --limit_num_examples 10 \
    --cache_clip_similarity /nethome/bdevnani3/flash/lmms_eval_cache/clip_similarity/videomme

accelerate launch \
    --num_processes=1 \
    --main_process_port 1111 \
    -m lmms_eval  \
    --model llava_onevision \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-ov" \
    --tasks longvideobench_test_v \
    --batch_size 1 \
    --output_path /nethome/bdevnani3/flash/.cache/results/llava_onevision/videomme/${SLURM_JOB_ID} \
    --log_samples \
    --im_resize_shape 16 \
    --max_frames_num 30 \
    --limit_num_examples 20 